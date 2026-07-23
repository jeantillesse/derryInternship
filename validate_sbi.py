import os
import sys
import json
import shutil
import pickle
import numpy as np
import scipy.interpolate
import torch
import matplotlib.pyplot as plt

# Enable Julia multithreading before loading juliacall and handle signals to prevent segfaults
os.environ["PYTHON_JULIACALL_THREADS"] = "auto"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

# Juliacall setup to avoid lock issues
if not (os.environ.get("PYTHON_JULIACALL_EXE") and os.environ.get("PYTHON_JULIACALL_PROJECT")):
    prefix = sys.prefix if sys.prefix != sys.base_prefix else os.getenv("CONDA_PREFIX")
    if prefix:
        project = os.path.abspath(os.path.join(prefix, "julia_env"))
    else:
        project = os.path.abspath(os.path.join(os.path.expanduser("~"), ".julia", "environments", "pyjuliapkg"))
    
    meta_path = os.path.join(project, "pyjuliapkg", "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            exe = meta.get("executable")
            if exe and shutil.which(exe):
                os.environ["PYTHON_JULIACALL_EXE"] = shutil.which(exe)
                os.environ["PYTHON_JULIACALL_PROJECT"] = project
        except Exception:
            pass

from juliacall import Main as jl

print("Loading Julia dependencies...")
jl.seval('using Pkg; Pkg.activate(".")')
jl.seval("using JLD2, FileIO, Statistics")
jl.include("passerelle.jl")
jl.include("tigaret_curves.jl")

print("Loading Tigaret data...")
donnees_brutes = jl.load("Tigaret_data.jld2", "alldata")

plot_order_indices = [4, 6, 0, 5, 2, 1, 3]
cibles = []
for idx in plot_order_indices:
    protocole = donnees_brutes[idx]
    valeurs = np.array(protocole)
    moyenne = np.mean(valeurs)
    variance = np.var(valeurs) # variance de population
    cibles.extend([moyenne, variance])

x_observe = torch.tensor(cibles, dtype=torch.float32)

file_save_name = "modele_entraine_box.pkl"
if not os.path.exists(file_save_name):
    print(f"Modèle {file_save_name} introuvable.")
    sys.exit(1)

print("Loading trained model...")
with open(file_save_name, "rb") as f:
    posterior = pickle.load(f)

num_samples = 50
print(f"Drawing {num_samples} highly probable samples from the peak of the posterior...")

# Tirer beaucoup d'échantillons pour trouver la zone la plus dense (le sommet de la cloche)
large_batch_size = 5000
many_samples = posterior.sample((large_batch_size,), x=x_observe, show_progress_bars=False)

# Calculer la probabilité (log_prob) de chaque échantillon
log_probs = posterior.log_prob(many_samples, x=x_observe)

# Garder uniquement les échantillons avec la plus haute probabilité (le sommet de la distribution)
top_indices = torch.topk(log_probs, num_samples).indices
echantillons = many_samples[top_indices]

# Define parallel computation function in Julia to bypass Python GIL
jl.seval("""
const MAPPING_K = [4, 3, 8, 5, 6, 7, 2]

function compute_everything_parallel(py_params_list; n_sweeps=10)
    native_params = [Vector{Float64}(p) for p in py_params_list]
    num_samples = length(native_params)
    
    tasks = []
    # k va de 1 à 7 pour les 7 protocoles
    for k in 1:7
        for sweep in 1:n_sweeps
            push!(tasks, (k, 0, [120.0, 15.0, 3.0, 0.2, 80000.0, 13000.0], sweep)) # Base model
            for i in 1:num_samples
                push!(tasks, (k, i, native_params[i], sweep))
            end
        end
    end
    
    # Storage for results (Float64 for delta_W)
    all_samples = [[Vector{Float64}(undef, n_sweeps) for _ in 1:num_samples] for _ in 1:7]
    all_base = [Vector{Float64}(undef, n_sweeps) for _ in 1:7]
    
    total_tasks = length(tasks)
    completed = Base.Threads.Atomic{Int}(0)
    
    println("Lancement de la simulation de $total_tasks tâches (evolveSynapse_light, $n_sweeps sweeps) sur $(Threads.nthreads()) cœurs Julia...")
    
    Threads.@threads for task in tasks
        k, i, p, sweep = task
        val_ampa, val_nmda, val_ca, val_neck, val_K_D, val_K_P = p[1], p[2], p[3], p[4], p[5], p[6]
        mapped_k = MAPPING_K[k]
        
        val_caT = val_ca
        val_caR = val_ca
        val_caL = val_ca
        
        delta_W = simuler_synapse_brute(val_ampa, val_nmda, val_caT, val_caR, val_caL, val_neck, val_K_D, val_K_P, mapped_k)
        
        if i == 0
            all_base[k][sweep] = delta_W
        else
            all_samples[k][i][sweep] = delta_W
        end
        
        c = Base.Threads.atomic_add!(completed, 1) + 1
        if c % 20 == 0 || c == total_tasks
            print("\rAvancement : $c / $total_tasks tâches terminées...   ")
        end
    end
    println()
    
    # Filter out NaNs
    samples_results = [Vector{Vector{Float64}}(undef, num_samples) for _ in 1:7]
    base_results = Vector{Vector{Float64}}(undef, 7)
    
    for k in 1:7
        valid_base = filter(x -> !isnan(x), all_base[k])
        base_results[k] = valid_base
        
        for i in 1:num_samples
            valid_samp = filter(x -> !isnan(x), all_samples[k][i])
            samples_results[k][i] = valid_samp
        end
    end
    
    return samples_results, base_results
end
""")

print(f"\nOptimization Phase: Evaluating {num_samples} candidates on the true simulator (2 sweeps each)...")
params_list = [t.numpy().tolist() for t in echantillons]
samples_results, _ = jl.compute_everything_parallel(params_list, n_sweeps=2)

best_mse = float('inf')
best_idx = 0

for i in range(num_samples):
    mse = 0
    valid = True
    for k in range(7):
        res = samples_results[k][i]
        if len(res) == 0:
            valid = False
            break
        m = np.mean(res)
        target_m = cibles[2*k]
        mse += (m - target_m)**2
    
    if not valid:
        continue
        
    if mse < best_mse:
        best_mse = mse
        best_idx = i

print(f"Meilleur candidat trouvé (index {best_idx}) avec MSE = {best_mse:.2f}")

# Keep only the best sample
best_sample = echantillons[best_idx]
echantillons = best_sample.unsqueeze(0)

print("\n--- Paramètres optimaux choisis par l'IA ---")
labels = ["N_ampa", "N_nmda", "N_ca*", "L_neck", "K_D", "K_P"]
for nom, val in zip(labels, best_sample):
    print(f"  {nom} = {val.item():.4f}")
print("-----------------------------------\n")

print("Running final simulation (10 sweeps) and plotting...")
params_list = [best_sample.numpy().tolist()]
n_sweeps = 10
samples_results, base_results = jl.compute_everything_parallel(params_list, n_sweeps=n_sweeps)

fig, axes = plt.subplots(1, 7, figsize=(24, 6), sharey=False)

MAPPING_K_PY = [4, 3, 8, 5, 6, 7, 2]

for k in range(1, 8):
    ax = axes[k-1]
    mapped_k = MAPPING_K_PY[k-1]
    proto_name = str(jl.seval(f"DATA_PROTOCOL[{mapped_k}, :protocol]"))
    
    ax.set_title(proto_name, fontsize=12, fontweight='bold')
    if k == 1:
        ax.set_ylabel("Weight Change (%)", fontsize=14)
    ax.axhline(0, color='gray', linestyle='--')
    
    # --- 1. Plot Base Model (Green) ---
    def_data = np.array(base_results[k-1])
    if len(def_data) > 0:
        w_pct_def = (def_data - 1.0) * 100.0
        # Draw a violin plot
        parts = ax.violinplot(w_pct_def, positions=[1], showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor('green')
            pc.set_alpha(0.5)
        for partname in ('cbars','cmins','cmaxes','cmeans'):
            vp = parts[partname]
            vp.set_edgecolor('green')
            vp.set_linewidth(2)
            
    # --- 2. Plot Posterior Samples (Blue) ---
    julia_samples = samples_results[k-1]
    all_sbi_pct = []
    for i, data in enumerate(julia_samples):
        if data is not None and len(data) > 0:
            w_pct_s = (np.array(data) - 1.0) * 100.0
            all_sbi_pct.extend(w_pct_s)
            
    if len(all_sbi_pct) > 0:
        parts = ax.violinplot(all_sbi_pct, positions=[2], showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor('blue')
            pc.set_alpha(0.5)
        for partname in ('cbars','cmins','cmaxes','cmeans'):
            vp = parts[partname]
            vp.set_edgecolor('blue')
            vp.set_linewidth(2)
            
    # --- 3. Plot Experimental Data (Red) ---
    exp_data = np.array(donnees_brutes[plot_order_indices[k-1]])
    exp_pct = (exp_data - 1.0) * 100.0
    exp_mean_pct = np.mean(exp_pct)
    exp_std_pct = np.std(exp_pct)
    
    ax.errorbar([1.5], [exp_mean_pct], yerr=[exp_std_pct], fmt='o', color='red', capsize=5, zorder=6, markersize=8)
    
    # Plot individual raw data points
    jitter = np.random.normal(1.5, 0.05, size=len(exp_pct))
    ax.scatter(jitter, exp_pct, color='gray', edgecolor='black', zorder=5, s=30)
    
    ax.set_xticks([1, 1.5, 2])
    ax.set_xticklabels(['Base', 'Data', 'SBI'])

plt.tight_layout()
plt.savefig("validation_distributions.png", dpi=150)
print("\\nValidation distributions saved as 'validation_distributions.png'.")
