## Imports
from math import log, e
import pickle
import random
import csv
from hashlib import sha256
import mmh3
from tqdm import tqdm
import time
import numpy as np

## Constants
N = 2 ** 16 # keyspace
p = 1 - e ** -2 # our table coverage - 86%
t = round(log(1-p)/log(1-N**(-1/3))) # chain length t
alphas = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] # different alpha values to try

## Functions to build cherry tables
def H_c(x):
    return sha256(x.to_bytes(8, 'little')).digest()  # return bytes directly

def r_c(N, t, y, i, ell=0):
    seed = int(i) + (ell*t)
    return mmh3.hash(y, seed, signed=False) % N

def build_cherry_table(N, t, startpoints, Kis):

    # parameters to store how many hashes and reductions are done, as well as actual time it takes to make this table
    hashes = 0
    reductions = 0
    duration = 0

    # get parameters for reduction function search
    t_bits = (t-1).bit_length() # number of bits to represent t
    k_bits = 32 - t_bits  # number of bits to represent rf index
    max_k_index = 2**k_bits  # maximum number of reduction functions per column

    # record # points per column
    points_per_column = []
    points_per_column.append(len(startpoints))

    # record rf samples list
    rf_results = []

    ##############################################################################################################

    # start monitoring time
    start = time.perf_counter()


    # store table in dictionary
    # initialise the table with sp:sp pairs
    table = {sp: sp for sp in startpoints}  # store all the points then remove duplicate entries - can't do duplicate keys in dictionary anyway so we can just store all ep:sp

    # Instead of making the table chain by chain, we have to make it column by column to test what reduction function to choose
    # store reduction function indexes
    rf_indexes = []

    # for each column
    for i in tqdm(range(t), desc=f"Calculating columns: "):

        # variable to store best cherry-pick
        best_trial = -1

        # hash all current points then store with startpoints - this stores all our current points
        hashed_points = {H_c(mi): sp for mi, sp in table.items()}
        hashes += (len(startpoints))    # increment hashes count

        # we are going to continuously replace table with the best rf trial, so we empty it for now
            # we haven't lost the current points as we have them hashed in the hashed_points dictionary
        table = {}

        # get # cherry-picks for this column
        k_i = round(Kis[i])

        # get indexes to trial
        pick_sample = np.array(random.sample(range(max_k_index), k_i), dtype=np.uint32)
        index_sample = (i << k_bits) | pick_sample

        # record what rf gives us
        rf_samples = [int(x) for x in index_sample]
        rf_points = []

        # trial all the reduction functions for this column
        for rf_trial in index_sample:

            # create a trial column to store results of current trial
            trial_column = {}

            # go through each key in hashed_points and store its reduction with sp
            for x in hashed_points:
                # reduce the hash and store in column
                trial_column[r_c(N, t, x, rf_trial)] = hashed_points[x]
                reductions += 1

            # store how many points this rf trial got
            rf_points.append(len(trial_column))

            # if trial_column is bigger than current table stored, we replace it
            if len(trial_column) > len(table):
                # replace it 
                table = trial_column
                # replace best cherry-pick
                best_trial = rf_trial

        # now store the best rf cherry pick
        rf_indexes.append(best_trial)
        # store # points in this column
        points_per_column.append(len(table))
        # store rf results for this column
        rf_results.append([rf_samples, rf_points])


    # finished making table so stop recording time
    duration = time.perf_counter() - start

    # finished, so return table and rf indexes
    return table, rf_indexes, hashes, reductions, duration, points_per_column, rf_results

## Get Kis for different alphas
optimisation_kis = []
optimisation_mt = []
optimisation_costs = []
with open("cherry_N_16_different_alpha_0.5_0.95_optimisation_results.pkl", "rb") as f:
    data = pickle.load(f)

for item in data:
    optimisation_kis.append(item[0])
    optimisation_mt.append(item[3][-1])
    optimisation_costs.append(item[2])

## Start Runs
# build the tables using the results
samples = 20
generation_results = []
generation_mt = []
generation_costs = []
generation_ppc = []
generation_rf_results = []

for a, alpha in enumerate(alphas):
    # build the actual table - sample times
    # first generate starpoints and store them
    mt_max = (2*N)/(t+2)
    mt_target = alpha * mt_max
    cherry_m_0 = round(mt_target/(1-alpha))    # our starting m_0

    # one set of startpoints for this alpha
    startpoints = random.sample(range(0, N-1), k=cherry_m_0)  # list to hold starpoint sets for this alpha

    ##############################################################

    # build the cherry table sample times and store it for later
    this_alpha_tables = []
    this_alpha_mt = []
    this_alpha_costs = []
    this_alpha_ppc = []
    this_alpha_rf_results = []

    # get the optimisation kis for this alpha
    optimisation_kis_for_alpha = optimisation_kis[a]

    # build sample tables
    for i in range(samples):
        # make the table with these startpoints
        table, rf_indexes, hashes, reductions, duration, points_per_column, rf_results = build_cherry_table(N, t, startpoints, optimisation_kis_for_alpha)
        # append to this alpha's tables
        this_alpha_tables.append((table, rf_indexes))
        this_alpha_mt.append(len(table))
        this_alpha_costs.append(hashes + (reductions * 1/577.44))  # total cost is hashes + reductions * 1/577.44
        this_alpha_ppc.append(points_per_column)
        this_alpha_rf_results.append(rf_results)
    
    # store this tables rf results in a pickle file
    with open(f"trials_data/cherry_N_16_alpha_{alpha}_rf_results.pkl", "wb") as f:
        pickle.dump(this_alpha_rf_results, f)


    # append to main list
    generation_results.append(this_alpha_tables)
    generation_mt.append(this_alpha_mt)
    generation_costs.append(this_alpha_costs)
    generation_ppc.append(this_alpha_ppc)   
    generation_rf_results.append(this_alpha_rf_results)

###################################################################

# store costs and mt in a csv file
with open("trials_data/cherry_N_16_20_trials_alpha_0.5_0.95_mt_cost_results.csv", "w", newline='') as f:
    f_alpha = csv.writer(f) # create a csv writer

    # create a list of headers for the different trials
    trial_headers = []
    cost_headers = []
    for i in range(samples):
        trial_headers.append(f"trial_{i+1}")
        cost_headers.append(f"cost_{i+1}")
    
    f_alpha.writerow(["alpha", "optimiser_mt"] + trial_headers  + ["optimiser_cost"] + cost_headers)

    # add in data for each alpha
    for i in range(len(alphas)):
        row = [alphas[i], optimisation_mt[i]] + generation_mt[i] + [optimisation_costs[i]] + generation_costs[i]
        f_alpha.writerow(row)

###################################################################

# store points per column in a csv file
for alpha in alphas:
    with open(f"trials_data/cherry_N_16_20_trials_alpha_{alpha}_points_per_column.csv", "w", newline='') as f:
        f_alpha = csv.writer(f) # create a csv writer

        # create a list of headers for the different trials
        column_headers = []
        for i in range(t+1):
            column_headers.append(f"Column_{i+1}")
        
        f_alpha.writerow(["Trial"] + column_headers)

        # add in data for each trial
        alpha_index = alphas.index(alpha)
        for trial in range(samples):
            row = [f"Trial_{trial+1}"] + generation_ppc[alpha_index][trial]
            f_alpha.writerow(row)

