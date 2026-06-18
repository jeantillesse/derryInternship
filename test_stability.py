import os
import sys
import json
import shutil

os.environ["JULIA_NUM_THREADS"] = "auto"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

# Contourner la résolution automatique de juliapkg pour éviter les verrous de fichiers
# concurrents sur le système de fichiers partagé (NFS/Lustre) du cluster.
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
import numpy as np
import torch

# Include bridge file
jl.include("passerelle.jl")

def run_test_params(params):
    # params is a list of [N_ampa, N_nmda, N_ca*, L_neck]
    try:
        # Running with multithread=True to use all 12 cores
        res = jl.simulateur_complet_sbi(params, multithread=True, nb_protocoles=7, n_sweeps=2)
        has_nan = any(np.isnan(val) for val in res)
        return "SUCCESS" if not has_nan else "NAN_IN_RESULTS"
    except Exception as e:
        return f"CRASH: {str(e)}"

# We want to test different parameter values:
test_cases = [
    # Baseline
    [120.0, 15.0, 3.0, 0.2],
    # Low values
    [0.0001, 15.0, 3.0, 0.2],
    [120.0, 0.0001, 3.0, 0.2],
    [120.0, 15.0, 0.0001, 0.2],
    [120.0, 15.0, 3.0, 0.0001],
    # High values
    [200.0, 15.0, 3.0, 0.2],
    [120.0, 20.0, 3.0, 0.2],
    [120.0, 15.0, 10.0, 0.2],
    [120.0, 15.0, 3.0, 2.0],
    # Combinations of high values
    [120.0, 20.0, 10.0, 0.2], 
    [120.0, 20.0, 10.0, 0.05], 
    [120.0, 20.0, 10.0, 2.0],
]

print("Running stability tests with multithreading...")
for tc in test_cases:
    status = run_test_params(tc)
    print(f"Params {tc} -> {status}")
