import os
import sys
import json
import shutil
import pickle
import numpy as np

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
import torch
import numpy as np
import matplotlib.pyplot as plt

print("Loading Julia dependencies...")
jl.seval("using JLD2, FileIO, Statistics")
jl.include("passerelle.jl")
jl.include("tigaret_curves.jl")

print("Loading Tigaret data...")
donnees_brutes = jl.load("Tigaret_data.jld2", "alldata")

cibles = []
for protocole in donnees_brutes[:7]:
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

num_samples = 5
print(f"Drawing {num_samples} samples from the posterior...")
echantillons = posterior.sample((num_samples,), x=x_observe)

print("Running simulations and plotting...")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

# Les noms des protocoles (k de 2 à 8 dans TigaretMellor16, qui correspondent à donnees_brutes[0:7])
for k in range(1,8):
    ax = axes[k-1]
    
    # Eval with string concatenation to avoid BoundsError
    proto_name = str(jl.seval(f"DATA_PROTOCOL[{k}, :protocol]"))
    
    ax.set_title(proto_name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Weight Change")
    
    max_t = 0.0
    
    for i, t in enumerate(echantillons):
        params = t.numpy().tolist()
        if len(params) == 4:
            val_ampa, val_nmda, val_ca, val_neck = params
            val_caT = val_caR = val_caL = val_ca
        else:
            val_ampa, val_nmda, val_caT, val_caR, val_caL, val_neck = params
            val_ca = val_caT
            
        # Call simuler_synapse_courbe
        # Actually, in tigaret_curves.jl, it uses `simuler_synapse_courbe(..., k)` where k is the index in DATA_PROTOCOL.
        # But wait, in passerelle.jl, it iterates `for k in 1:nb_protocoles` and calls `simuler_synapse_brute(..., k)` which uses `DATA_PROTOCOL[k, ...]`.
        try:
            curve = jl.simuler_synapse_courbe(val_ampa, val_nmda, val_ca, val_neck, k)
            t_sec = np.array(curve.t)
            
            # curve.weight_change is LTP - LTD (raw number of receptors).
            # Convert to percentage relative to N_ampa to match Tigaret data format:
            weight_change_pct = (np.array(curve.weight_change) / val_ampa) * 100.0
            
            max_t = max(max_t, t_sec[-1])
            label = "Posterior Samples" if (i == 0 or (i > 0 and len(ax.lines) == 0)) else ""
            ax.plot(t_sec, weight_change_pct, color="blue", alpha=0.3, label=label)
        except Exception as e:
            # Handle solver instabilities
            print(f"Sample {i} failed for protocol {k} due to instability.")
            
    # Simulate and plot with Base Model Default Parameters
    val_ampa_def, val_nmda_def, val_ca_def, val_neck_def = 120.0, 15.0, 3.0, 0.2
    try:
        curve_def = jl.simuler_synapse_courbe(val_ampa_def, val_nmda_def, val_ca_def, val_neck_def, k)
        t_sec_def = np.array(curve_def.t)
        weight_change_pct_def = (np.array(curve_def.weight_change) / val_ampa_def) * 100.0
        
        max_t = max(max_t, t_sec_def[-1])
        ax.plot(t_sec_def, weight_change_pct_def, color="green", linewidth=2, label="Base Model (Default)")
    except Exception as e:
        print(f"Base model default parameters failed for protocol {k}.")
        
    # Plot experimental data point at the end
    exp_data = np.array(donnees_brutes[k-1])
    # Experimental data is normalized (e.g. 1.2 = 120%). Convert to percent change (20%).
    exp_mean_pct = (np.mean(exp_data) - 1.0) * 100.0
    exp_std_pct = np.std(exp_data) * 100.0
    
    ax.errorbar([max_t], [exp_mean_pct], yerr=[exp_std_pct], fmt='o', color='red', capsize=5, label="Tigaret Data (Mean ± STD)")
    ax.legend()

plt.tight_layout()
plt.savefig("validation_plot.png", dpi=300)
print("Plot saved to validation_plot.png!")
