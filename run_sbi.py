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
    
    indices_protocoles = [1, 2, 3, 4, 5, 6, 7] 
    # Nombre de répétitions par protocole pour capter le bruit (variance)
    N_SWEEPS = 10 

    for jeu_de_parametres in theta:
        val_ampa = float(jeu_de_parametres[0])
        val_nmda = float(jeu_de_parametres[1])
        val_vgcc = float(jeu_de_parametres[2])
        val_neck = float(jeu_de_parametres[3])
        
        vecteur_cible = [] 
        
        # 2. Boucle sur les 7 protocoles de Tigaret
        for k_protocole in indices_protocoles:
            changements_de_poids = []
            
            # 3. Boucle des répétitions stochastiques
            for _ in range(N_SWEEPS):
                delta_w = jl.simuler_synapse_brute(val_ampa, val_nmda, val_vgcc, val_neck, k_protocole)
                changements_de_poids.append(float(delta_w))
            
            # 4. Calcul de la moyenne et de la variance
            moyenne_dw = np.mean(changements_de_poids)
            variance_dw = np.var(changements_de_poids)
            
            vecteur_cible.extend([moyenne_dw, variance_dw])
            
        courbe_pytorch = torch.tensor(vecteur_cible, dtype=torch.float32)
        resultats_simules.append(courbe_pytorch)
        
    return torch.stack(resultats_simules)


# --- PROTOCOLE MACHINE LEARNING (SBI) ---

# Ordre : [N_ampa, N_nmda, N_vgcc, L_neck]
prior = BoxUniform(
    low=torch.tensor([10.0, 5.0, 0.0, 0.05]), 
    high=torch.tensor([300.0, 50.0, 20.0, 1.5]) 
)

print("\nLancement des simulations de l'IA...")
theta_sim, x_sim = simulate_for_sbi(simulateur_sbi_hybride, prior, num_simulations=100, show_progress_bar=True)

print("Entraînement du réseau de neurones...")
inference = NPE(prior)
inference.append_simulations(theta_sim, x_sim)
inference.train()
posterior = inference.build_posterior()

print("Interface validée ! L'IA est entraînée.\n")

# --- ÉTAPE FINALE : INFÉRENCE AVEC LES VRAIES DONNÉES ---

print("Chargement des données expérimentales depuis Tigaret_data.jld2...")

jl.seval('using JLD2, FileIO')
donnees_brutes = jl.load("Tigaret_data.jld2", "alldata")

# 2. Calcul dynamique de la moyenne et de la variance
cibles = []
for protocole in donnees_brutes:
    valeurs = np.array(protocole)
    
    moyenne = np.mean(valeurs)
    variance = np.var(valeurs) # Variance (ddof=0 par défaut dans numpy)
    
    cibles.extend([moyenne, variance])

x_observe = torch.tensor(cibles, dtype=torch.float32)

print("Tenseur cible x_observe calculé avec succès :")
print(x_observe)

echantillons = posterior.sample((10000,), x=x_observe)

print("\nGénération terminée ! Vous pouvez maintenant analyser les résultats.")

fig, axes = pairplot(echantillons, labels=["N_ampa", "N_nmda", "N_vgcc", "L_neck"])
plt.show()