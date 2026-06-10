import argparse
import os
import pickle
import torch
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Générateur de simulations en parallèle (Job Array)")
    parser.add_argument("--task_id", type=int, required=True, help="ID de la tâche SLURM (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--nb_sims", type=int, default=10, help="Nombre de simulations à générer pour cette tâche")
    parser.add_argument("--gauss", action="store_true", help="Le prior est une gausienne")
    parser.add_argument("--box", action="store_true", help="Le prior est une Box uniforme")
    parser.add_argument("--nb_protocoles", type=int, default=7, choices=range(1, 8), help="Nombre de protocoles à tester (1 à 7)")
    parser.add_argument("--n_sweeps", type=int, default=10, help="Nombre de sweeps par protocole")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Configuration des graines aléatoires uniques
    base_seed = 42000
    unique_seed = base_seed + args.task_id
    
    torch.manual_seed(unique_seed)
    np.random.seed(unique_seed)
    
    # Configuration des variables d'environnement pour Julia
    os.environ["JULIA_NUM_THREADS"] = "auto"
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    
    # 2. Importer JuliaCall et initialiser le seed Julia
    from juliacall import Main as jl
    jl.seval(f"using Random; Random.seed!({unique_seed})")
    
    # Charger les modules et fonctions Julia
    jl.include("sbi/passerelle.jl")
    
    from sbi.utils import BoxUniform
    from torch.distributions import MultivariateNormal, TransformedDistribution, ExpTransform
    
    # 3. Définir le prior
    if not args.gauss and not args.box:
        args.box = True
        
    if args.box:
        prior = BoxUniform(
            low=torch.tensor([5, 0.5, 0.2, 0.02]), 
            high=torch.tensor([200.0, 18.0, 8.0, 1.6]) 
        )
    elif args.gauss:
        moyennes_log = torch.log(torch.tensor([100.0, 10.0, 5.0, 1.0]))
        ecarts_types_log = torch.tensor([0.3, 0.3, 0.3, 0.3])
        matrice_covariance_log = torch.diag(ecarts_types_log ** 2)
        base_dist = MultivariateNormal(loc=moyennes_log, covariance_matrix=matrice_covariance_log)
        prior = TransformedDistribution(base_dist, ExpTransform())

    # Stratégie de parallélisation Julia interne (multi-threading sur le lot)
    # On utilise ParallelStrategy car cpus-per-task > 1
    USE_MULTITHREAD = True
    
    def simulateur_sbi_hybride(theta, start_sim_idx=1):
        theta = torch.atleast_2d(theta)
        batch_params = [[float(p) for p in jeu_de_parametres] for jeu_de_parametres in theta]
        strategy = jl.ParallelStrategy() if USE_MULTITHREAD else jl.SequentialStrategy()
        
        resultats_julia = jl.simulateur_sbi(
            strategy, 
            batch_params, 
            nb_protocoles=args.nb_protocoles, 
            n_sweeps=args.n_sweeps, 
            start_sim_idx=start_sim_idx
        )
        
        resultats_simules = []
        for res in resultats_julia:
            courbe_pytorch = torch.tensor(list(res), dtype=torch.float32)
            resultats_simules.append(courbe_pytorch)
        return torch.stack(resultats_simules)

    from run_sbi import filtrer_parametres

    print(f"\n[Task {args.task_id}] Lancement de {args.nb_sims} simulations (Seed: {unique_seed})...")
    theta_sim_list = []
    x_sim_list = []
    tentatives = 0
    batch_size = 12
    
    while len(x_sim_list) < args.nb_sims:
        needed = args.nb_sims - len(x_sim_list)
        current_batch_size = min(batch_size, needed)
        
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
        
        x_batch = simulateur_sbi_hybride(theta_batch, start_sim_idx=len(x_sim_list) + 1)
        
        for t, x in zip(theta_batch, x_batch):
            if not torch.isnan(x).any() and not torch.isinf(x).any():
                theta_sim_list.append(t.unsqueeze(0))
                x_sim_list.append(x.unsqueeze(0))
                print(f"➜ [Task {args.task_id}] Sim {len(x_sim_list)}/{args.nb_sims} réussie (tentatives: {tentatives}).")
            
    theta_sim = torch.cat(theta_sim_list, dim=0)
    x_sim = torch.cat(x_sim_list, dim=0)

    # Sauvegarder les résultats temporaires
    os.makedirs("sbi/sims_temp", exist_ok=True)
    out_file = f"sbi/sims_temp/sim_job_{args.task_id}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump((theta_sim, x_sim), f)
        
    print(f"✓ [Task {args.task_id}] Sauvegardé avec succès dans {out_file}\n")

if __name__ == "__main__":
    main()
