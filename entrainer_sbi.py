import argparse
import glob
import os
import pickle
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform
from torch.distributions import MultivariateNormal, TransformedDistribution, ExpTransform

def get_args():
    parser = argparse.ArgumentParser(description="Script pour fusionner les simulations et entraîner le modèle SBI")
    parser.add_argument("--gauss", action="store_true", help="Le prior est une gausienne")
    parser.add_argument("--box", action="store_true", help="Le prior est une Box uniforme")
    return parser.parse_args()

def main():
    args = get_args()
    
    if not args.gauss and not args.box:
        args.box = True
        
    if args.box:
        file_save_name = "modele_entraine_box"
        prior = BoxUniform(
            low=torch.tensor([5, 0.5, 0.2, 0.02]), 
            high=torch.tensor([200.0, 18.0, 8.0, 1.6]) 
        )
    elif args.gauss:
        file_save_name = "modele_entraine_gauss"
        moyennes_log = torch.log(torch.tensor([100.0, 10.0, 5.0, 1.0]))
        ecarts_types_log = torch.tensor([0.3, 0.3, 0.3, 0.3])
        matrice_covariance_log = torch.diag(ecarts_types_log ** 2)
        base_dist = MultivariateNormal(loc=moyennes_log, covariance_matrix=matrice_covariance_log)
        prior = TransformedDistribution(base_dist, ExpTransform())

    print("Fusion des résultats des simulations temporaires...")
    all_thetas = []
    all_xs = []

    fichiers = glob.glob("sbi/sims_temp/sim_job_*.pkl")
    print(f"Trouvé {len(fichiers)} fichiers de simulation.")

    if len(fichiers) == 0:
        print("Erreur : Aucun fichier temporaire trouvé dans sbi/sims_temp/")
        return

    for f_path in fichiers:
        try:
            with open(f_path, "rb") as f:
                theta, x = pickle.load(f)
                all_thetas.append(theta)
                all_xs.append(x)
        except Exception as e:
            print(f"Avertissement : Impossible de lire {f_path} ({e}), fichier ignoré.")

    theta_total = torch.cat(all_thetas, dim=0)
    x_total = torch.cat(all_xs, dim=0)

    print(f"Taille totale des simulations fusionnées : {x_total.shape}")
    print("Entraînement du réseau de neurones NPE...")

    inference = NPE(prior)
    inference.append_simulations(theta_total, x_total)
    inference.train()
    posterior = inference.build_posterior()

    # Sauvegarde du modèle final dans le dossier sbi
    with open(f"sbi/{file_save_name}.pkl", "wb") as f:
        pickle.dump(posterior, f)

    print(f"✓ Modèle SBI entraîné avec succès et sauvegardé dans 'sbi/{file_save_name}.pkl' !")

if __name__ == "__main__":
    main()
