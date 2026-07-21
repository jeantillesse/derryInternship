#!/bin/bash
#SBATCH --job-name=sbi_array
#SBATCH --output=train_model/sims_temp/logs/sbi_%A_%a.out
#SBATCH --error=train_model/sims_temp/logs/sbi_%A_%a.err
#SBATCH --array=1-84                  # 84 tâches pour faire 1008 simulations (84 * 12)
#SBATCH --time=03:00:00               # 1h max (les 12 simulations prennent ~30 min avec 128 cœurs)
#SBATCH --partition=k2-hipri          # File haute priorité (maximum 3h)
#SBATCH --nodes=1                     # 1 nœud complet par tâche
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128           # Utilise les 128 cœurs du nœud pour Julia (multi-thread)
#SBATCH --mem=256G                    # Demande 256 Go de RAM sur le nœud (très large pour 128 cœurs)

# Créer les dossiers pour les résultats et les logs
mkdir -p train_model/sims_temp/logs

# Charger le module anaconda
module purge
module load apps/anaconda3/2024.10/bin

# Activer proprement l'environnement conda env_sbi
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env_sbi

# Résoudre l'erreur 'GLIBCXX_3.4.26 not found' (version de libstdc++ requise par Julia)
# On force le système à chercher d'abord dans les bibliothèques du Conda env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Désactiver le locking et la résolution de juliapkg pour le Job Array
# afin d'éviter les erreurs 'Stale file handle' (accès concurrents sur NFS/Lustre)
export PYTHON_JULIAPKG_OFFLINE=yes
export PYTHON_JULIAPKG_LOCKFILE=no

# Détermination du RUN_ID unique basé sur la date et l'heure de soumission du job
if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    # Récupère l'heure de soumission du job via scontrol
    SUBMIT_TIME=$(scontrol show job "$SLURM_ARRAY_JOB_ID" | grep -o 'SubmitTime=[^ ]*' | cut -d= -f2 | head -n 1)
    if [ -n "$SUBMIT_TIME" ] && [ "$SUBMIT_TIME" != "Unknown" ]; then
        # Format: YYYY-MM-DD_HHhMMmSS (ex: 2026-06-11_16h45m30)
        RUN_ID=$(echo "$SUBMIT_TIME" | sed 's/T/_/; s/:/h/; s/:/m/')
    else
        RUN_ID="job_${SLURM_ARRAY_JOB_ID}"
    fi
else
    # Fallback hors SLURM
    RUN_ID=$(date +%Y-%m-%d_%Hh%Mm%Ss)
fi

echo "Lancement du Job Array tâche n° ${SLURM_ARRAY_TASK_ID} sur le nœud ${SLURMD_NODENAME} avec 128 cœurs (RUN_ID: $RUN_ID)"

# Lancer la génération de 12 simulations (soit 840 simulations unitaires)
python generer_sims.py --task_id ${SLURM_ARRAY_TASK_ID} --nb_sims 12 --box --nb_protocoles 7 --n_sweeps 10 --run_id "$RUN_ID"
