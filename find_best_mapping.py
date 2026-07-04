import os
import sys
import json
import shutil
import pickle
import numpy as np
import itertools
from scipy.optimize import linear_sum_assignment

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

exp_means = []
for protocole in donnees_brutes[:7]:
    exp_data = np.array(protocole)
    exp_mean_pct = (np.mean(exp_data) - 1.0) * 100.0
    exp_means.append(exp_mean_pct)

print("Experimental means (targets):", exp_means)

print("Running simulations...")
jl.seval("""
function get_base_model_curves()
    results = []
    # indices 2 to 8
    for k in 2:8
        try
            curve = simuler_synapse_courbe(120.0, 15.0, 3.0, 0.2, k)
            push!(results, (collect(curve.t), collect(curve.weight_change)))
        catch e
            push!(results, nothing)
        end
    end
    return results
end
""")

base_results = jl.get_base_model_curves()
sim_means = []

for i, res in enumerate(base_results):
    if res is not None:
        w_pct = np.array(res[1]) * 100.0
        final_w = w_pct[-1]
    else:
        final_w = np.nan
    sim_means.append(final_w)
    print(f"Simulation for protocol index {i+2}: final weight change = {final_w}")

cost_matrix = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        # We can use absolute error or squared error
        cost_matrix[i, j] = abs(exp_means[i] - sim_means[j])

row_ind, col_ind = linear_sum_assignment(cost_matrix)

best_mapping = [int(col_ind[i] + 2) for i in range(7)]

print("Best mapping for MAPPING_K_PY:", best_mapping)
