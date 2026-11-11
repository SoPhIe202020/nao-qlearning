"""
POC Q-learning (sans modèle) : le NAO apprend seul à atteindre une cible.

- État   : (distance à la cible discrétisée, angle relatif cible/robot discrétisé)
           calculés à partir des coordonnées (Supervisor).
- Actions: avancer, tourner à gauche 40°, tourner à droite 40° (fichiers .motion).
- Récompenses : +100 cible atteinte, -100 chute, -50 trop loin,
                -1 par pas + bonus proportionnel au rapprochement.
- Q-table tabulaire mise à jour avec Bellman :
      Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

Fichiers produits dans ce dossier :
  q_table.json        la Q-table apprise
  training_state.json numéro d'épisode courant (permet de reprendre)
  rewards.csv         récompense / nb de pas / succès par épisode (courbe d'apprentissage)
"""
from controller import Supervisor, Motion
import os
import math
import random
import json

# =============================
# CONFIGURATION
# =============================
MODE = "test"          # "train" pour apprendre, "test" pour voir la politique apprise (epsilon = 0)
MAX_EPISODES = 150
MAX_STEPS = 25          # actions max par épisode
TEST_EPISODES = 20

ALPHA = 0.2             # taux d'apprentissage
GAMMA = 0.9             # facteur d'actualisation
EPS_START = 0.9         # exploration au début
EPS_MIN = 0.05
EPS_DECAY = 0.97        # epsilon multiplié par ça à chaque épisode

GOAL_DIST = 0.35        # m : cible considérée atteinte
MAX_DIST = 3.0          # m : au-delà, l'épisode s'arrête (robot perdu)
RANDOMIZE = True        # orientation du robot et position de la cible aléatoires à chaque épisode

# Discrétisation de l'état
DIST_EDGES = [GOAL_DIST, 0.7, 1.2, 2.0]           # -> 5 cases (0 = cible atteinte)
ANGLE_EDGES = [-100, -45, -15, 15, 45, 100]        # degrés -> 7 cases

ACTIONS = ["forward", "turn_left", "turn_right"]

# =============================
# WEBOTS SETUP
# =============================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

nao_node = robot.getFromDef("NAO")
target_node = robot.getFromDef("TARGET")
if nao_node is None or target_node is None:
    raise SystemExit("ERREUR : il faut DEF NAO et DEF TARGET dans le monde Webots.")

nao_trans = nao_node.getField("translation")
nao_rot = nao_node.getField("rotation")
target_trans = target_node.getField("translation")

HERE = os.path.dirname(os.path.abspath(__file__))
Q_PATH = os.path.join(HERE, "q_table.json")
STATE_PATH = os.path.join(HERE, "training_state.json")
CSV_PATH = os.path.join(HERE, "rewards.csv")

# Position de départ = celle du monde au chargement
INIT_POS = list(nao_trans.getSFVec3f())
INIT_ROT = list(nao_rot.getSFRotation())
INIT_TARGET = list(target_trans.getSFVec3f())

# =============================
# MOUVEMENTS (.motion)
# =============================
MOTION_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "motions"))

def load_motion(name):
    path = os.path.join(MOTION_DIR, name)
    if not os.path.isfile(path):
        raise SystemExit("Motion introuvable : " + path)
    return Motion(path)

MOTIONS = {
    "forward": load_motion("Forwards.motion"),
    "turn_left": load_motion("TurnLeft40.motion"),
    "turn_right": load_motion("TurnRight40.motion"),
}

def step_for(ms):
    for _ in range(max(1, int(ms / timestep))):
        if robot.step(timestep) == -1:
            raise SystemExit(0)

def play_motion(motion):
    """Joue le mouvement JUSQU'AU BOUT (sinon le NAO reste dans une posture instable)."""
    motion.setTime(0)
    motion.play()
    try:
        max_steps = int(motion.getDuration() / timestep) + 25
    except Exception:
        max_steps = 400
    for _ in range(max_steps):
        if robot.step(timestep) == -1:
            raise SystemExit(0)
        if motion.isOver():
            break
    motion.stop()

# =============================
# GÉOMÉTRIE (monde NUE : Y = vertical, sol = plan X-Z)
# =============================
def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def nao_xz():
    p = nao_trans.getSFVec3f()
    return p[0], p[2]

def target_xz():
    p = target_trans.getSFVec3f()
    return p[0], p[2]

def heading_deg():
    """Direction 'devant' du NAO projetée au sol (axe X local du robot)."""
    R = nao_node.getOrientation()          # matrice 3x3 en ligne
    fx, fz = R[0], R[6]
    return math.degrees(math.atan2(fz, fx))

def is_fallen():
    R = nao_node.getOrientation()
    up_y = R[5]                            # composante verticale de l'axe Z local (haut du robot)
    return up_y < 0.6 or nao_trans.getSFVec3f()[1] < 0.22

def distance_and_angle():
    nx, nz = nao_xz()
    tx, tz = target_xz()
    dx, dz = tx - nx, tz - nz
    dist = math.hypot(dx, dz)
    angle = wrap_deg(math.degrees(math.atan2(dz, dx)) - heading_deg())
    return dist, angle

def bin_index(value, edges):
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)

def get_state():
    dist, angle = distance_and_angle()
    return (bin_index(dist, DIST_EDGES), bin_index(angle, ANGLE_EDGES))

# --- rotations (quaternions) pour tourner le robot autour de la verticale ---
def axis_angle_to_quat(r):
    x, y, z, a = r
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    s = math.sin(a / 2)
    return (math.cos(a / 2), x / n * s, y / n * s, z / n * s)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)

def quat_to_axis_angle(q):
    w, x, y, z = q
    w = max(-1.0, min(1.0, w))
    a = 2 * math.acos(w)
    s = math.sqrt(max(1e-12, 1 - w * w))
    return [x / s, y / s, z / s, a]

def rotation_with_yaw(yaw_rad):
    q_yaw = (math.cos(yaw_rad / 2), 0.0, math.sin(yaw_rad / 2), 0.0)   # autour de Y (vertical)
    return quat_to_axis_angle(quat_mul(q_yaw, axis_angle_to_quat(INIT_ROT)))

def reset_episode():
    if RANDOMIZE:
        yaw = random.uniform(-math.pi, math.pi)
        d = random.uniform(1.0, 1.8)
        th = random.uniform(-math.pi, math.pi)
        target = [INIT_POS[0] + d * math.cos(th), INIT_TARGET[1], INIT_POS[2] + d * math.sin(th)]
    else:
        yaw = 0.0
        target = INIT_TARGET
    nao_trans.setSFVec3f(INIT_POS)
    nao_rot.setSFRotation(rotation_with_yaw(yaw))
    target_trans.setSFVec3f(target)
    nao_node.resetPhysics()
    step_for(300)

# =============================
# Q-TABLE & PERSISTANCE
# =============================
def load_q():
    if os.path.isfile(Q_PATH):
        try:
            with open(Q_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {tuple(int(v) for v in k.split(",")): vals for k, vals in raw.items()}
        except Exception as e:
            print("Q-table illisible, on repart de zéro :", e)
    return {}

def save_q(q):
    with open(Q_PATH, "w", encoding="utf-8") as f:
        json.dump({"%d,%d" % k: v for k, v in sorted(q.items())}, f, indent=2)

def load_episode():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return int(json.load(f).get("episode", 1))
    except Exception:
        return 1

def save_episode(ep):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"episode": ep}, f)

def log_episode(ep, eps, total, steps, outcome):
    new = not os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", encoding="utf-8") as f:
        if new:
            f.write("episode,epsilon,total_reward,steps,outcome\n")
        f.write("%d,%.3f,%.2f,%d,%s\n" % (ep, eps, total, steps, outcome))

def q_values(q, s):
    if s not in q:
        q[s] = [0.0] * len(ACTIONS)
    return q[s]

def choose_action(q, s, eps):
    if random.random() < eps:
        return random.randrange(len(ACTIONS))
    qv = q_values(q, s)
    best = max(qv)
    return random.choice([i for i, v in enumerate(qv) if v == best])   # égalités départagées au hasard

# =============================
# ÉPISODE
# =============================
def run_episode(q, eps, learn):
    reset_episode()
    dist, _ = distance_and_angle()
    total, outcome, steps = 0.0, "timeout", 0
    for step in range(MAX_STEPS):
        s = get_state()
        a = choose_action(q, s, eps)
        play_motion(MOTIONS[ACTIONS[a]])
        steps = step + 1

        new_dist, new_angle = distance_and_angle()
        s2 = get_state()
        r = -1.0 + 20.0 * (dist - new_dist)        # coût du pas + bonus de rapprochement
        done = False
        if is_fallen():
            r, done, outcome = -100.0, True, "fall"
        elif new_dist < GOAL_DIST:
            r += 100.0
            done, outcome = True, "goal"
        elif new_dist > MAX_DIST:
            r -= 50.0
            done, outcome = True, "lost"

        if learn:
            target = r if done else r + GAMMA * max(q_values(q, s2))
            qv = q_values(q, s)
            qv[a] += ALPHA * (target - qv[a])

        total += r
        print("  pas %2d  état=%s  action=%-10s  dist=%.2f  angle=%+5.0f°  r=%+.1f"
              % (steps, s, ACTIONS[a], new_dist, new_angle, r))
        dist = new_dist
        if done:
            break
    return total, steps, outcome

# =============================
# MAIN
# =============================
q = load_q()

if MODE == "test":
    print("=== MODE TEST : politique apprise, epsilon = 0 ===")
    wins = 0
    for i in range(TEST_EPISODES):
        total, steps, outcome = run_episode(q, 0.0, learn=False)
        wins += outcome == "goal"
        print("Test %d : %s en %d pas (R=%.1f)" % (i + 1, outcome, steps, total))
        test_csv = os.path.join(HERE, "test_results.csv")
        first = not os.path.isfile(test_csv)
        with open(test_csv, "a", encoding="utf-8") as f:
            if first:
                f.write("test,total_reward,steps,outcome\n")
            f.write("%d,%.2f,%d,%s\n" % (i + 1, total, steps, outcome))
    print("Succès : %d/%d" % (wins, TEST_EPISODES))
    robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
else:
    episode = load_episode()
    print("=== ENTRAÎNEMENT Q-learning : reprise à l'épisode %d ===" % episode)
    while episode <= MAX_EPISODES:
        eps = max(EPS_MIN, EPS_START * EPS_DECAY ** (episode - 1))
        print("\n=== Épisode %d/%d  (epsilon=%.2f) ===" % (episode, MAX_EPISODES, eps))
        total, steps, outcome = run_episode(q, eps, learn=True)
        print("Fin épisode %d : %s en %d pas, récompense totale %.1f" % (episode, outcome, steps, total))
        save_q(q)
        log_episode(episode, eps, total, steps, outcome)
        episode += 1
        save_episode(episode)
        if outcome == "fall":
            # Après une chute, on recharge le monde pour retrouver un NAO debout ;
            # le contrôleur redémarre et reprend grâce aux fichiers sauvegardés.
            print("Chute -> rechargement du monde")
            robot.worldReload()
            step_for(1000)
    print("\n=== Entraînement terminé. Passe MODE = \"test\" pour voir le résultat. ===")
    robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
