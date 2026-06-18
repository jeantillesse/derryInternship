import os
os.environ["JULIA_NUM_THREADS"] = "auto"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"
os.environ["PYTHON_JULIAPKG_LOCKFILE"] = "no"
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
