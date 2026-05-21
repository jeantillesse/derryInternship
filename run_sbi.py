# =====================================================================
# Fichier : run_sbi.py
# C'est ici que sbi pilote Julia pour trouver les bons paramètres
# =====================================================================

import torch
import matplotlib.pyplot as plt
import numpy as np
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.analysis import pairplot

# --- CONNEXION À JULIA ---
print("Démarrage de Julia et chargement du modèle (patiente un peu)...")
from juliacall import Main as jl

# On demande à Julia de lire notre fichier passerelle
jl.include("passerelle.jl")
print("Pont Python-Julia établi avec succès !")


# --- DÉFINITION DU SIMULATEUR POUR SBI ---
def simulateur_sbi_hybride(theta):
    theta = torch.atleast_2d(theta)
    resultats_simules = []

    for jeu_de_parametres in theta:
        val_ampa = float(jeu_de_parametres[0])
        val_nmda = float(jeu_de_parametres[1])
        
        # NOUVEAU : On récupère le temps (t) et le voltage (v)
        t_julia, v_julia = jl.simuler_synapse_brute(val_ampa, val_nmda)
        
        # NOUVEAU : Interpolation sur une grille stricte de 500 points
        t_fixe = np.linspace(float(t_julia[0]), float(t_julia[-1]), 500)
        courbe_fixe = np.interp(t_fixe, list(t_julia), list(v_julia))
        
        # On donne la courbe lissée et fixe à PyTorch
        courbe_pytorch = torch.tensor(courbe_fixe, dtype=torch.float32)
        resultats_simules.append(courbe_pytorch)
        
    return torch.stack(resultats_simules)


# --- PROTOCOLE MACHINE LEARNING (SBI) ---

# Définition des bornes de recherche (Prior)
# ex: conductance_ampa entre 1.0 et 50.0, delai_pre entre 10.0 et 100.0 ms
prior = BoxUniform(low=torch.tensor([1.0, 10.0]), high=torch.tensor([50.0, 100.0]))

print("\nLancement des simulations de l'IA (Julia tourne en arrière-plan)...")
# On fait 100 simulations pour tester (tu augmenteras à 1000 ou 2000 plus tard)
theta_sim, x_sim = simulate_for_sbi(simulateur_sbi_hybride, prior, num_simulations=100)

print("Entraînement du réseau de neurones...")
inference = NPE(prior)
inference.append_simulations(theta_sim, x_sim)
inference.train()
posterior = inference.build_posterior()

print("Interface validée ! L'IA est entraînée et prête à l'inférence.")