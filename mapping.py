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


print("Running simulations and plotting...")

# Define parallel computation function in Julia to bypass Python GIL
# We MUST convert python lists to native Julia Vectors BEFORE entering the multithreading loop
jl.seval("""
const MAPPING_K = [4, 3, 8, 5, 6, 7, 2]

function compute_everything_parallel(py_params_list)
    native_params = [Vector{Float64}(p) for p in py_params_list]
    num_samples = length(native_params)
    
    tasks = []
    # k va de 1 à 7 pour les 7 protocoles
    for k in 1:7
        push!(tasks, (k, 0, [120.0, 15.0, 3.0, 0.2])) # Base model
        for i in 1:num_samples
            push!(tasks, (k, i, native_params[i]))
        end
    end
    
    samples_results = [Vector{Any}(undef, num_samples) for _ in 1:7]
    base_results = Vector{Any}(undef, 7)
    
    println("Lancement de la simulation parallèle sur $(Threads.nthreads()) cœurs Julia...")
    
    Threads.@threads for task in tasks
        k, i, p = task
        # 'p' est maintenant un Vector{Float64} 100% natif Julia : aucun appel à Python !
        val_ampa, val_nmda, val_ca, val_neck = p[1], p[2], p[3], p[4]
        mapped_k = MAPPING_K[k]
        try
            curve = simuler_synapse_courbe(val_ampa, val_nmda, val_ca, val_neck, mapped_k)
            # Store data as native tuples of arrays
            res = (collect(curve.t), collect(curve.weight_change))
            if i == 0
                base_results[k] = res
            else
                samples_results[k][i] = res
            end
        catch e
            if i == 0
                base_results[k] = nothing
            else
                samples_results[k][i] = nothing
            end
        end
    end
    
    return samples_results, base_results
end
""")

params_list = []

print("Running simulations in parallel via native Julia Threads...")
samples_results, base_results = jl.compute_everything_parallel(params_list)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

MAPPING_K_PY = [4, 3, 8, 5, 6, 7, 2]

for k in range(1, 8):
    ax = axes[k-1]
    mapped_k = MAPPING_K_PY[k-1]
    proto_name = str(jl.seval(f"DATA_PROTOCOL[{mapped_k}, :protocol]"))
    
    ax.set_title(proto_name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Weight Change (%)")
    
    max_t = 0.0
    interpolated_curves = []
    
    # Process samples
    valid_samples = 0
    valid_curves_data = []
    

    if max_t == 0: max_t = 1.0
    common_t = np.linspace(0, max_t, 500)
    
    for t_sec, w_pct in valid_curves_data:
        interp_func = scipy.interpolate.interp1d(t_sec, w_pct, bounds_error=False, fill_value=(w_pct[0], w_pct[-1]))
        interpolated_curves.append(interp_func(common_t))

    y_min_target = 0
    y_max_target = 0
    
    if interpolated_curves:
        median_curve = np.median(interpolated_curves, axis=0)
        ax.plot(common_t, median_curve, color="blue", linewidth=2.5, label="Median Posterior")
        y_min_target = np.min(median_curve)
        y_max_target = np.max(median_curve)
        
    # Process base model
    def_data = base_results[k-1]
    if def_data is not None:
        t_sec_def = np.array(def_data[0])
        w_raw_def = np.array(def_data[1])
        val_ampa_def = 120.0
        w_pct_def = w_raw_def * 100.0
        max_t = max(max_t, t_sec_def[-1])
        
        ax.plot(t_sec_def, w_pct_def, color="green", linewidth=2.5, label="Base Model (Default)")
        y_min_target = min(y_min_target, np.min(w_pct_def))
        y_max_target = max(y_max_target, np.max(w_pct_def))

    # Experimental data
    exp_data = np.array(donnees_brutes[plot_order_indices[k-1]])
    exp_mean_pct = (np.mean(exp_data) - 1.0) * 100.0
    exp_std_pct = np.std(exp_data) * 100.0
    ax.errorbar([max_t], [exp_mean_pct], yerr=[exp_std_pct], fmt='o', color='red', capsize=5, label="Tigaret Data")
    
    y_min_target = min(y_min_target, exp_mean_pct - exp_std_pct)
    y_max_target = max(y_max_target, exp_mean_pct + exp_std_pct)
    
    margin = (y_max_target - y_min_target) * 0.15
    if margin == 0: margin = 10
    ax.set_ylim([y_min_target - margin, y_max_target + margin])
    
    ax.legend(loc='lower left', fontsize='small')

plt.tight_layout()
plt.savefig("validation_plot.png", dpi=300)
print("Plot saved to validation_plot.png!")
