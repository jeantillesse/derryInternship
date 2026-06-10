import argparse
import pickle
import os

os.environ["JULIA_NUM_THREADS"] = "auto"

os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
# 2. IMPORTER JULIACALL EN TOUT PREMIER (AVANT Torch et Numpy)
from juliacall import Main as jl

print(f"🔥 Nombre de threads Julia actifs : {jl.Threads.nthreads()}")

# 3. Seulement maintenant, importer le reste
import torch
from torch.distributions import MultivariateNormal, TransformedDistribution, ExpTransform
import numpy as np
import matplotlib.pyplot as plt
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.analysis import pairplot


USE_MULTITHREAD = True
NB_PROTOCOLES = 7
N_SWEEPS = 10

def get_args():
    parser = argparse.ArgumentParser(description="Script SBI pour SynapseElife")
    parser.add_argument("--train", action="store_true", help="Lancer les simulations et entraîner le modèle")
    parser.add_argument("--test", action="store_true", help="Charger le modèle sauvegardé et faire l'inférence")
    parser.add_argument("--gauss", action="store_true", help="Le prior est une gausienne")
    parser.add_argument("--box", action="store_true", help="Le prior est une Box uniforme")
    parser.add_argument("--singlethread", action="store_true", help="Désactiver le multi-threading dans Julia")
    parser.add_argument("--nb_protocoles", type=int, default=7, choices=range(1, 8), help="Nombre de protocoles à tester (1 à 7)")
    parser.add_argument("--n_sweeps", type=int, default=10, help="Nombre de sweeps par protocole (valeur par défaut : 10)")
    return parser.parse_args()

jl.include("passerelle.jl")

def simulateur_sbi_hybride(theta, start_sim_idx=1):
    theta = torch.atleast_2d(theta)
    
    # Conversion du lot (batch) de paramètres en liste Python pour Julia
    batch_params = [[float(p) for p in jeu_de_parametres] for jeu_de_parametres in theta]
    
    # Sélection de la stratégie d'exécution en Julia
    strategy = jl.ParallelStrategy() if USE_MULTITHREAD else jl.SequentialStrategy()
    
    # Appel du simulateur Strategy-based en Julia avec l'index de départ
    resultats_julia = jl.simulateur_sbi(strategy, batch_params, nb_protocoles=NB_PROTOCOLES, n_sweeps=N_SWEEPS, start_sim_idx=start_sim_idx)
    
    resultats_simules = []
    for res in resultats_julia:
        courbe_pytorch = torch.tensor(list(res), dtype=torch.float32)
        resultats_simules.append(courbe_pytorch)
        
    import gc
    gc.collect()
        
    return torch.stack(resultats_simules)


def filtrer_parametres(theta):
    """
    Filtre les paramètres d'entrée en amont de la simulation pour écarter
    les configurations instables ou aberrantes (stiffness ODE ou LTD infinie).
    """
    params = theta.flatten().tolist()
    
    # Ordre des paramètres : [N_ampa, N_nmda, N_ca*, L_neck]
    n_ampa, n_nmda, n_ca, l_neck = params
    
    # 1. Protection contre les valeurs trop faibles (évite LTD infinie ou division par zéro)
    if n_ampa < 5.0 or n_nmda < 0.5 or n_ca < 0.2 or l_neck < 0.02:
        return False
        
    # 2. Protection contre les valeurs individuelles trop élevées (évite la stiffness thermique/calcique)
    if n_nmda > 18.0 or n_ca > 8.0 or l_neck > 1.6:
        return False
        
    # 3. Éviter les combinaisons critiques de charge calcique accumulée :
    # Plus N_nmda et N_ca sont élevés, et plus L_neck est grand (diffusion lente),
    # plus la raideur du système d'équations biochimiques explose.
    charge_calcique = (n_nmda + 2.0 * n_ca) * l_neck
    if charge_calcique > 22.0:
        return False
        
    return True


def main():
    global USE_MULTITHREAD, NB_PROTOCOLES, N_SWEEPS
    args = get_args()

    NB_PROTOCOLES = args.nb_protocoles
    N_SWEEPS = args.n_sweeps

    if args.singlethread:
        USE_MULTITHREAD = False

    # Si aucun argument n'est fourni, on fait les deux par défaut
    if not args.train and not args.test:
        args.train = True
        args.test = True

    if not args.gauss and not args.box:
        args.box = True
    if args.box:
        file_save_name = "modele_entraine_box"
    elif args.gauss:
        file_save_name = "modele_entraine_gauss"

    # --- PROTOCOLE MACHINE LEARNING (SBI) ---
    if args.box:
        # Ordre : [N_ampa, N_nmda, N_ca*, L_neck]
        prior = BoxUniform(
            low=torch.tensor([5, 0.5, 0.2, 0.02]), 
            high=torch.tensor([200.0, 18.0, 8.0, 1.6]) 
        )
    elif args.gauss:

        moyennes_log = torch.log(torch.tensor([100.0, 10.0, 5.0, 1.0]))
        ecarts_types_log = torch.tensor([0.3, 0.3, 0.3, 0.3]) # exemple d'écarts-types en espace log
        matrice_covariance_log = torch.diag(ecarts_types_log ** 2)
        base_dist = MultivariateNormal(loc=moyennes_log, covariance_matrix=matrice_covariance_log)
        # Le prior génère des échantillons strictement positifs
        prior = TransformedDistribution(base_dist, ExpTransform())

    if args.train:
        print("\nLancement des simulations de l'IA (avec rejet automatique des échecs et instabilités)...")
        theta_sim_list = []
        x_sim_list = []
        num_simulations = 100
        tentatives = 0
        
        batch_size = 12  # Taille du lot (batch) de paramètres simulés en parallèle
        while len(x_sim_list) < num_simulations:
            needed = num_simulations - len(x_sim_list)
            current_batch_size = min(batch_size, needed)
            
            # On génère plus de candidats pour compenser le filtrage en amont
            candidats = prior.sample((current_batch_size * 4,))
            valides = []
            for t in candidats:
                if filtrer_parametres(t.unsqueeze(0)):
                    valides.append(t)
                    if len(valides) == current_batch_size:
                        break
            
            if len(valides) == 0:
                continue
                
            theta_batch = torch.stack(valides)
            tentatives += len(theta_batch)
            
            # Simulation en parallèle du lot en passant l'index de départ
            x_batch = simulateur_sbi_hybride(theta_batch, start_sim_idx=len(x_sim_list) + 1)
            
            for t, x in zip(theta_batch, x_batch):
                if not torch.isnan(x).any() and not torch.isinf(x).any():
                    theta_sim_list.append(t.unsqueeze(0))
                    x_sim_list.append(x.unsqueeze(0))
                    print(f"➜ Simulation {len(x_sim_list)}/{num_simulations} réussie (total tentatives: {tentatives}).")
                
        theta_sim = torch.cat(theta_sim_list, dim=0)
        x_sim = torch.cat(x_sim_list, dim=0)

        print("\n--- DEBUG INFO ---")
        print("Taille de x_sim :", x_sim.shape)
        print("Aperçu des 5 premières lignes de x_sim :", x_sim[:5])
        print("Nombre total de NaN :", torch.isnan(x_sim).sum().item())
        print("Nombre total de Inf :", torch.isinf(x_sim).sum().item())
        print("------------------\n")

        print("Entraînement du réseau de neurones...")
        inference = NPE(prior)
        inference.append_simulations(theta_sim, x_sim)
        inference.train()
        posterior = inference.build_posterior()

        with open(f"{file_save_name}.pkl", "wb") as f:
            pickle.dump(posterior, f)

        print(f"Interface validée ! L'IA est entraînée et le modèle est sauvegardé dans '{file_save_name}.pkl'.\n")

    # --- ÉTAPE FINALE : INFÉRENCE AVEC LES VRAIES DONNÉES ---

    if args.test:
        print("\n--- ÉTAPE FINALE : INFÉRENCE AVEC LES VRAIES DONNÉES ---")
        
        if not args.train:
            if not os.path.exists(f"{file_save_name}.pkl"):
                print(f"Erreur : Le modèle '{file_save_name}.pkl' est introuvable. Veuillez d'abord l'entraîner avec l'option --train.")
                exit(1)
            with open(f"{file_save_name}.pkl", "rb") as f:
                posterior = pickle.load(f)
            print(f"Modèle chargé avec succès depuis '{file_save_name}.pkl'.")

        print("Chargement des données expérimentales depuis Tigaret_data.jld2...")

        jl.seval('using JLD2, FileIO')
        donnees_brutes = jl.load("Tigaret_data.jld2", "alldata")

        # 2. Calcul dynamique de la moyenne et de la variance
        cibles = []
        for protocole in donnees_brutes[:args.nb_protocoles]:
            valeurs = np.array(protocole)
            
            moyenne = np.mean(valeurs)
            variance = np.var(valeurs) # Variance (ddof=0 par défaut dans numpy)
            
            cibles.extend([moyenne, variance])

        x_observe = torch.tensor(cibles, dtype=torch.float32)

        print("Tenseur cible x_observe calculé avec succès :")
        print(x_observe)

        echantillons = posterior.sample((10000,), x=x_observe)

        print("\nGénération terminée ! Vous pouvez maintenant analyser les résultats.")

        valeurs_par_defaut = torch.tensor([120.0, 15.0, 3.0, 0.2])
        fig, axes = pairplot(echantillons, labels=["N_ampa", "N_nmda", "N_ca*", "L_neck"], points=valeurs_par_defaut, points_colors=["red"])
        plt.show()

if __name__ == "__main__":
    main()