
import torch
import matplotlib.pyplot as plt
import numpy as np
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.analysis import pairplot

# --- CONNEXION À JULIA ---
from juliacall import Main as jl

jl.include("passerelle.jl")


# --- DÉFINITION DU SIMULATEUR POUR SBI ---
def simulateur_sbi_hybride(theta):
    theta = torch.atleast_2d(theta)
    resultats_simules = []

    for jeu_de_parametres in theta:
        val_ampa = float(jeu_de_parametres[0])
        val_nmda = float(jeu_de_parametres[1])
        
        t_julia, v_julia = jl.simuler_synapse_brute(val_ampa, val_nmda)
        
        t_fixe = np.linspace(float(t_julia[0]), float(t_julia[-1]), 500)
        courbe_fixe = np.interp(t_fixe, list(t_julia), list(v_julia))
        
        courbe_pytorch = torch.tensor(courbe_fixe, dtype=torch.float32)
        resultats_simules.append(courbe_pytorch)
        
    return torch.stack(resultats_simules)


# --- PROTOCOLE MACHINE LEARNING (SBI) ---

prior = BoxUniform(low=torch.tensor([1.0, 10.0]), high=torch.tensor([50.0, 100.0]))

print("\nLancement des simulations de l'IA")
theta_sim, x_sim = simulate_for_sbi(simulateur_sbi_hybride, prior, num_simulations=100)

print("Entraînement du réseau de neurones...")
inference = NPE(prior)
inference.append_simulations(theta_sim, x_sim)
inference.train()
posterior = inference.build_posterior()

print("Interface validée ! L'IA est entraînée et prête à l'inférence.")