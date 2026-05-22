
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

prior = BoxUniform(
    low=torch.tensor([0.001, 0.001]), 
    high=torch.tensor([0.1, 0.1])
)

print("\nLancement des simulations de l'IA")
theta_sim, x_sim = simulate_for_sbi(simulateur_sbi_hybride, prior, num_simulations=100)

print("Entraînement du réseau de neurones...")
inference = NPE(prior)
inference.append_simulations(theta_sim, x_sim)
inference.train()
posterior = inference.build_posterior()

print("Interface validée ! L'IA est entraînée et prête à l'inférence.")


# 2. LE TEST DU JUMEAU NUMÉRIQUE
print("\n--- TEST SCIENTIFIQUE ---")
# On choisit des valeurs "secrètes" très différentes du défaut pour voir si l'IA les trouve
vrai_ampa = 0.08
vrai_nmda = 0.02
print(f"Génération de l'expérience secrète (AMPA={vrai_ampa}, NMDA={vrai_nmda})...")
t_cible, v_cible = jl.simuler_synapse_brute(vrai_ampa, vrai_nmda)

t_fixe = np.linspace(float(t_cible[0]), float(t_cible[-1]), 500)
courbe_cible = torch.tensor(np.interp(t_fixe, list(t_cible), list(v_cible)), dtype=torch.float32)

print("L'IA tente de retrouver les valeurs secrètes...")
echantillons_devines = posterior.sample((10000,), x=courbe_cible)

fig, axes = pairplot(
    echantillons_devines,
    limits=[[0.001, 0.1], [0.001, 0.1]], 
    labels=["Conductance AMPA", "Conductance NMDA"],
    points=[[vrai_ampa, vrai_nmda]], # Ajoute une croix rouge sur les VRAIES valeurs !
    points_colors=["red"]
)
plt.show()