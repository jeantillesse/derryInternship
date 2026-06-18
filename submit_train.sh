#!/bin/bash
#SBATCH --job-name=sbi_train
#SBATCH --output=train_model/train_sbi.out
#SBATCH --error=train_model/train_sbi.err
#SBATCH --time=00:30:00               # 30 minutes suffisent largement pour l'entraînement
#SBATCH --partition=k2-hipri          # File haute priorité
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # 4 cœurs pour l'entraînement PyTorch
#SBATCH --mem-per-cpu=2G

# Charger le module anaconda
module purge
module load apps/anaconda3/2024.10/bin

# Activer proprement l'environnement conda env_sbi
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env_sbi

# Résoudre l'erreur 'GLIBCXX_3.4.26 not found' (version de libstdc++ requise par Julia)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Début de l'entraînement SBI..."

# Lancer l'entraînement sur toutes les simulations générées
python entrainer_sbi.py --box
