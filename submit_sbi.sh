#!/bin/bash
#SBATCH --job-name=sbi_array
#SBATCH --output=sbi/sims_temp/logs/sbi_%A_%a.out
#SBATCH --error=sbi/sims_temp/logs/sbi_%A_%a.err
#SBATCH --array=1-84                  # 84 tâches pour faire 1008 simulations (84 * 12)
#SBATCH --time=01:00:00               # 1h max (les 12 simulations prennent ~30 min avec 128 cœurs)
#SBATCH --partition=k2-hipri          # File haute priorité (maximum 3h)
#SBATCH --nodes=1                     # 1 nœud complet par tâche
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128           # Utilise les 128 cœurs du nœud pour Julia (multi-thread)
#SBATCH --mem=256G                    # Demande 256 Go de RAM sur le nœud (très large pour 128 cœurs)

# Créer les dossiers pour les résultats et les logs
mkdir -p sbi/sims_temp/logs

# Charger l'environnement conda
module load apps/anaconda3/2024.06/bin


source activate env_sbi
# Activer votre environnement s'il y en a un
# source activate env_sbi

echo "Lancement du Job Array tâche n° ${SLURM_ARRAY_TASK_ID} sur le nœud ${SLURMD_NODENAME} avec 128 cœurs"

# Lancer la génération de 12 simulations (soit 840 simulations unitaires)
python generer_sims.py --task_id ${SLURM_ARRAY_TASK_ID} --nb_sims 12 --box --nb_protocoles 7 --n_sweeps 10
