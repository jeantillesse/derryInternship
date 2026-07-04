import os
import sys
import json
import shutil
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

# Handle signals and threading for Julia
os.environ["PYTHON_JULIACALL_THREADS"] = "auto"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

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

print("Loading Tigaret experimental data...")
donnees_brutes = jl.load("Tigaret_data.jld2", "alldata")

# Re-define the parallel execution function for just the base model
jl.seval("""
const MAPPING_K = [8, 7, 6, 2, 4, 5, 3]

function compute_base_model(; n_sweeps=20)
    tasks = []
    # k va de 1 à 7 pour les 7 protocoles de donnees_brutes
    for k in 1:7
        for sweep in 1:n_sweeps
            push!(tasks, (k, sweep))
        end
    end
    
    # On stocke tous les sweeps pour chaque protocole
    all_results = [Vector{Any}(undef, n_sweeps) for _ in 1:7]
    
    Threads.@threads for task in tasks
        k, sweep = task
        mapped_k = MAPPING_K[k]
        val_ampa = 120.0
        val_nmda = 15.0
        val_ca = 3.0
        val_neck = 0.2
        try
            curve = simuler_synapse_courbe(val_ampa, val_nmda, val_ca, val_neck, mapped_k)
            all_results[k][sweep] = (collect(curve.t), collect(curve.weight_change))
        catch e
            all_results[k][sweep] = nothing
        end
    end
    
    # Faire la moyenne de tous les sweeps pour avoir la courbe "moyenne"
    base_results = Vector{Any}(undef, 7)
    for k in 1:7
        sweeps = all_results[k]
        valid_sweeps = filter(x -> x !== nothing, sweeps)
        if isempty(valid_sweeps)
            base_results[k] = nothing
        else
            # On suppose que tous les sweeps ont le même vecteur temps (t) 
            # (car les temps sont fixés par l'intégrateur ou l'interpolation,
            # mais si les vecteurs temps varient, il faut interpoler. 
            # Dans SynapseElife, evolveSynapse_light sort des temps variables si events... 
            # Attention, si `t` varie, on ne peut pas juste faire la moyenne des y.
            base_results[k] = valid_sweeps # On renvoie tous les sweeps valides, Python fera la médiane/moyenne
        end
    end
    
    return base_results
end
""")

print("Simulating Base Model with multiple sweeps (Stochastic averaging)...")
base_results = jl.compute_base_model(n_sweeps=5)

# L'ordre d'affichage désiré par la capture d'écran de votre maître de stage
# Les indices de 0 à 6 de donnees_brutes correspondent à:
# 0: 1Pre1Post10
# 1: 1Pre2Post10
# 2: 1Pre2Post50
# 3: 2Pre10
# 4: 2Post1Pre50
# 5: 2Post1Pre20
# 6: 2Pre50
# L'ordre sur la capture d'écran est:
# 2Post1Pre50, 2Pre50, 1Pre1Post10, 2Post1Pre20, 1Pre2Post50, 1Pre2Post10, 2Pre10
plot_order_indices = [4, 6, 0, 5, 2, 1, 3]

# On crée une grande figure
fig, axes = plt.subplots(1, 7, figsize=(24, 6), sharey=False)

MAPPING_K_PY = [8, 7, 6, 2, 4, 5, 3]

for subplot_idx, data_idx in enumerate(plot_order_indices):
    ax = axes[subplot_idx]
    k = data_idx + 1 # k va de 1 à 7
    mapped_k = MAPPING_K_PY[data_idx]
    
    proto_name = str(jl.seval(f"DATA_PROTOCOL[{mapped_k}, :protocol]"))
    
    ax.set_title(proto_name, fontsize=12, fontweight='bold')
    if subplot_idx == 0:
        ax.set_ylabel("Weight Change (%)", fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.axhline(0, color='gray', linestyle='--')
    
    # 1. Plot Base Model Simulation (Green)
    sweeps_data = base_results[data_idx]
    max_t = 0.0
    if sweeps_data is not None and len(sweeps_data) > 0:
        interpolated_sweeps = []
        common_t = None
        
        # Trouver le max_t global
        for sweep in sweeps_data:
            t_sec_sweep = np.array(sweep[0])
            max_t = max(max_t, t_sec_sweep[-1])
            
        common_t = np.linspace(0, max_t, 500)
        
        for sweep in sweeps_data:
            t_sec_sweep = np.array(sweep[0])
            w_raw_sweep = np.array(sweep[1])
            w_pct_sweep = w_raw_sweep * 100.0
            
            # Trace chaque sweep en clair
            ax.plot(t_sec_sweep, w_pct_sweep, color="green", alpha=0.1)
            
            interp_func = scipy.interpolate.interp1d(t_sec_sweep, w_pct_sweep, bounds_error=False, fill_value=(w_pct_sweep[0], w_pct_sweep[-1]), kind='previous')
            interpolated_sweeps.append(interp_func(common_t))
            
        # Moyenne des sweeps
        mean_curve = np.mean(interpolated_sweeps, axis=0)
        ax.plot(common_t, mean_curve, color="green", linewidth=3, label="Base Model (Mean)")
        
    # 2. Plot Experimental Data (Red dots + Error bars)
    exp_data = np.array(donnees_brutes[data_idx])
    exp_mean_pct = (np.mean(exp_data) - 1.0) * 100.0
    exp_std_pct = np.std(exp_data) * 100.0
    
    # Trace les points individuels de l'expérience (Scatter plot) comme dans le papier
    exp_data_pct = (exp_data - 1.0) * 100.0
    # On ajoute un léger "jitter" sur l'axe X (vers max_t) pour que les points ne se superposent pas
    jitter = np.random.normal(max_t, max_t * 0.02, size=len(exp_data_pct))
    
    ax.scatter(jitter, exp_data_pct, color='gray', edgecolor='black', zorder=5, label="Raw Data")
    
    # Trace la moyenne et l'écart-type
    ax.errorbar([max_t], [exp_mean_pct], yerr=[exp_std_pct], fmt='o', color='red', capsize=5, zorder=6, label="Data Mean ± STD")
    
    if subplot_idx == 0:
        ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig("base_model_vs_tigaret.png", dpi=150)
print("\nSuccess! The plot has been saved as 'base_model_vs_tigaret.png'.")
