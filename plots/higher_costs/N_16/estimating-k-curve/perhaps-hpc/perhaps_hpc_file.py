import os
import random
import csv
from utils import *
from math import e, log
import pickle

## Constants
N = 2 ** 20
p = 1-e**-2 # our table coverage
cost_factor = 5
alpha = 0.62  # maximality factor
t = round(log(1-p)/log(1-N**(-1/3)))

# how many total samples exist
samples = 20

# get this job's index (0-based)
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# read in pickle file containing all the kjs values
with open("cherry_N_20_alpha_0.62_cost_5_optimisation_results.pkl", "rb") as f:
        opt_res = pickle.load(f)
kjs, m_0, final_cost, m_values = opt_res

# ---- everything below now runs ONLY ONE sample ----

# compute parameters
mt_max = (2*N)/(t+2)
mt_target = alpha * mt_max
cherry_m_0 = round(mt_target/(1-alpha))

# generate starpoints for THIS sample only
starpoints = set()
while len(starpoints) < cherry_m_0:
    starpoints.add(random.randint(0, N-1))

# build table
optimisation_kis_for_alpha = kjs
table, rf_indexes, hashes, reductions, duration = build_cherry_table(
    N, t, starpoints, optimisation_kis_for_alpha
)

mt = len(table)
cost = hashes + (reductions * 1/577.44)

# write result (one file per sample to avoid clashes)
output_file = f"data/sample_{task_id}.csv"

with open(output_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id", "alpha", "mt", "cost"])
    writer.writerow([task_id, alpha, mt, cost])