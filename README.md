# SBI - Synapse Model Optimization

This folder contains the Python-Julia interface to optimize the SynapseElife model using Simulation-Based Inference (SBI).

## How to run

1. **Create a virtual environment:**
   `python3 -m venv env_sbi`
   `source env_sbi/bin/activate`

2. **Install requirements:**
   `python3 -m pip install torch sbi matplotlib juliacall numpy`

3. **Run the optimization:**
   `python3 run_sbi.py`
