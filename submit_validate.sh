#!/bin/bash
#SBATCH --job-name=val_sbi
#SBATCH --output=validate.out
#SBATCH --error=validate.err
#SBATCH --time=00:30:00               # Devrait prendre quelques minutes
#SBATCH --partition=k2-hipri          # File haute priorité
#SBATCH --nodes=1                     # 1 nœud complet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128           # Utilise les 128 cœurs du nœud
#SBATCH --mem=128G                    # RAM

# Charger le module anaconda
module purge
module load apps/anaconda3/2024.10/bin

# Activer l'environnement
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env_sbi

# Résoudre l'erreur GLIBCXX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Dire à Julia d'utiliser les 128 cœurs
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Lancement de la validation avec $JULIA_NUM_THREADS cœurs sur le nœud $SLURMD_NODENAME..."
python validate_sbi.py
echo "Validation terminée !"
