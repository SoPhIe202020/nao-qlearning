"""Ajoute à rewards.csv une colonne 'moyenne10' : moyenne glissante de la
récompense sur les 10 derniers épisodes. Produit rewards_lisse.csv,
lu par pgfplots dans le rapport et la présentation."""
import csv

with open("rewards.csv", encoding="utf-8") as f:
    lignes = list(csv.DictReader(f))

recompenses = []
for ligne in lignes:
    recompenses.append(float(ligne["total_reward"]))
    fenetre = recompenses[-10:]
    ligne["moyenne10"] = "%.2f" % (sum(fenetre) / len(fenetre))

with open("rewards_lisse.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
    w.writeheader()
    w.writerows(lignes)
