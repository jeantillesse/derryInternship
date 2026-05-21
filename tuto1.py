import torch
import matplotlib.pyplot as plt
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.analysis import pairplot


def simulateur_synapse(theta):
    # 1. On force theta à avoir 2 dimensions (Lignes=Simulations, Colonnes=Paramètres)
    theta = torch.atleast_2d(theta)
    
    # 2. On extrait A et tau pour TOUTES les simulations en même temps
    # La syntaxe [:, 0:1] veut dire "Toutes les lignes, et uniquement la colonne 0"
    A = theta[:, 0:1]    
    tau = theta[:, 1:2]  
    
    # 3. L'échelle de temps reste la même
    t = torch.linspace(0, 10, 50)
    
    # 4. Magie de PyTorch : il multiplie la colonne A par la ligne de temps t
    # Le résultat 'courant' aura automatiquement une taille (2000 lignes, 50 colonnes)
    courant = A * torch.exp(-t / tau)
    
    # 5. On génère un bruit de la même dimension que notre résultat
    bruit = torch.randn_like(courant) * 0.05
    
    return courant + bruit



# On cherche A entre 0.1 et 5.0, et tau entre 0.5 et 3.0
limite_basse = torch.tensor([0.1, 0.5])
limite_haute = torch.tensor([5.0, 3.0])

prior = BoxUniform(low=limite_basse, high=limite_haute)



print("Génération des simulations en cours...")
# La fonction simulate_for_sbi gère automatiquement la création du dataset
theta_simules, x_simules = simulate_for_sbi(simulateur_synapse, prior, num_simulations=2000)

print("Entraînement du réseau de neurones...")
inference = NPE(prior)
inference.append_simulations(theta_simules, x_simules)
estimateur_densite = inference.train()

# Création du modèle final (le posterior)
posterior = inference.build_posterior()



# 1. On invente une "vraie" donnée issue du labo (par exemple générée par A=3.0 et tau=1.5)
vrais_parametres = torch.tensor([3.0, 1.5])
donnee_experimentale = simulateur_synapse(vrais_parametres)

# 2. On demande à notre IA entraînée de deviner les paramètres !
echantillons_devines = posterior.sample((5000,), x=donnee_experimentale)

# 3. On affiche le résultat
fig, axes = pairplot(
    echantillons_devines,
    limits=[[0, 5], [0, 4]],
    labels=["Amplitude (A)", "Constante temps (tau)"],
    figsize=(6, 6)
)
plt.show()