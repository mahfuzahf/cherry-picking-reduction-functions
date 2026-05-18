from hashlib import sha256
import time
from tqdm import tqdm
import mmh3
from math import log, pi, sqrt
from statistics import NormalDist
from scipy.optimize import minimize, Bounds
from scipy.stats import norm
import pickle
import numpy as np
import random

###############################################################################################################################

# Hash function
def H(x):
	return int.from_bytes(sha256(x.to_bytes(8)).digest())

# Reduction function
# currently mod but should change to murmurhash in future
def r(N, t, y, i, ell=0):   # also takes in ell - number of tables - for future use (but currently ell=0)
	return (y + i + ell*t) % N

def H_c(x):
    return sha256(x.to_bytes(8, 'little')).digest()  # return bytes directly

def r_c(N, t, y, i, ell=0):
    seed = int(i) + (ell*t)
    return mmh3.hash(y, seed, signed=False) % N

###############################################################################################################################

# build cherry table - store how many hashes and reductions are done
# returns: cherry table, indexes, no. hashes, no. reductions, time to build
def build_cherry_table(N, t, startpoints, Kis):

    # parameters to store how many hashes and reductions are done, as well as actual time it takes to make this table
    hashes = 0
    reductions = 0
    duration = 0

    # get parameters for reduction function search
    t_bits = (t-1).bit_length() # number of bits to represent t
    k_bits = 32 - t_bits  # number of bits to represent rf index
    max_k_index = 2**k_bits  # maximum number of reduction functions per column

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

        # trial all the reduction functions for this column
        for rf_trial in index_sample:

            # create a trial column to store results of current trial
            trial_column = {}

            # go through each key in hashed_points and store its reduction with sp
            for x in hashed_points:
                # reduce the hash and store in column
                trial_column[r_c(N, t, x, rf_trial)] = hashed_points[x]
                reductions += 1

            # if trial_column is bigger than current table stored, we replace it
            if len(trial_column) > len(table):
                # replace it 
                table = trial_column
                # replace best cherry-pick
                best_trial = rf_trial

        # now store the best rf cherry pick
        rf_indexes.append(best_trial)


    # finished making table so stop recording time
    duration = time.perf_counter() - start

    # finished, so return table and rf indexes
    return table, rf_indexes, hashes, reductions, duration

# build vanilla table - store how many hashes and reductions are done
# returns: vanilla table, no. hashes, no. reductions
def build_vanilla_table_with_costs(N, t, startpoints, filename="False"):
    # store table in dictionary
    table = {}
    hashes = 0
    reductions = 0

    # for each startpoint
    for sp in startpoints:
        current_point = sp  # keep track of current point in chain -  we want to store startpoint later

        # create the chain
        for i in range(t):
            # hash then reduce the value
            current_point = r(N, t, H(current_point), i)
            hashes += 1
            reductions += 1

        # check if there wasn't a chain merge (not in a value stored already) - if not then store in table
        if current_point not in table:
            table[current_point] = sp

    # store the table as a pickle file
    if (filename != "False"):
        with open(filename, 'wb') as f:
            pickle.dump(table, f)

    return table, hashes, reductions
###############################################################################################################################

def m_t(inputs, N, m_0):  ## the m_t function as a target function
    ### the input variables to find are as follows:
    # input[0] is the number of starting points
    # input[1:] is the K_j for all 1<=j<=t

    K_js = inputs[1:]  # K_j values
    m = m_0  # starting m value

    # this is doing some sort of stats to find the expected m_i+1 value for the column 
    for k in K_js:  # iterate through the column
        # Calculate E1 and E2
        E1 = (1 - 1 / N) ** m
        E2 = (1 - 2 / N) ** m
        # calculate the statistical average of m_j+1
        average = N * (1 - E1)
        # check if the variance is positive, sometimes error happens because of float calculation on very small values
        if N * ((N - 1) * E2 + E1 - N * E1 ** 2) < 0:   # this equation gives the variance, so sqrt to get standard deviation
            # if it is negative (error), set it to 0
            sd = 0
        else:
            # count the statistical variance of m_j+1 otherwise
            sd = sqrt(N * ((N - 1) * E2 + E1 - N * E1 ** 2))
        # find the m_j+1
        m = average + NormalDist().inv_cdf((k - pi / 8) / (k - pi / 4 + 1)) * sd

    # return -m to maximise the amount of points with optimiser
    return -m


## define the objective and target functions
def cost(inputs, N, m_0):  ## the cost calculation function as objective function to be minimized
    # ### the input variables to find are as follows:
    # input[0] is the number of starting points
    # input[1:] is the K_j for all 1<=j<=t
    K_js = inputs[1:]  # K_j values
    cost = 0  # cost counter
    m_values = [m_0]  # list to keep the mj values

    for k in K_js:  # iterate through the column
        # Calculate E1 and E2
        E1 = (1 - 1 / N) ** m_0
        E2 = (1 - 2 / N) ** m_0
        # calculate the statistical average of m_j+1
        average = N * (1 - E1)
        ## check if the variance is positive, sometimes error happens because of float calculation on very small values
        if N * ((N - 1) * E2 + E1 - N * E1 ** 2) < 0:
            # if it is negative (error), set it to 0
            variance = 0
        else:
            # count the variance otherwise
            variance = sqrt(N * ((N - 1) * E2 + E1 - N * E1 ** 2))
        # count the cost before defining the new m
        cost += m_0 + m_0 * k / 30
        # find the m_j+1
        m_0 = average + NormalDist().inv_cdf((k - pi / 8) / (k - pi / 4 + 1)) * variance
        m_values.append(m_0)
        # count the cost

    # return cost - not square rooted as we are bounding cost
    return cost, m_values

def vanilla_cost(N, m_0, t):
    # # keep track of cost
    cost = 0
    # set m to be the starting m_0
    m = m_0
    # append m to the list of m_values to be returned
    m_values = [m]

    # iterate through each column
    for i in range(t):
        # calculate cost - add 1 for each hash we do (m hashes in this column)
        cost = cost + m

        # use formula m_i+1 = N(1 - (1 - 1/N)^m_i)
        m = N * (1 - ((1 - (1 / N)) ** m))
        m_values.append(m)

    return cost, m_values

###############################################################################################################################

# Optimise 
# add vcost to bound the optimisation
def optimise_kj(N, p, alpha, factor, lower_bound, t=-1, maxiter=5000000, eps=1):   # add v_cost if want vanilla cost to be set beforehand

    if t == -1:
        t = round(log(1-p)/log(1-N**(-1/3))) # Calculate t

    mt_max = (2*N)/(t+2)
    mt_target = alpha * mt_max
    if alpha == 1:
        m_0 = N
    else:
        m_0 = round(mt_target/(1-alpha))    # our starting m_0

    
    ## set the bound and target function
    bound = Bounds([lower_bound for i in range(t)], [float('inf') for i in range(t)])
    
    # our target function that needs to be > 0
    # instead of using the target function, we can use the cost function as the objective function
    # need to get the cost of a vanilla rainbow table with the same parameters
    
    # Const * P(vanilla) - P(cherry) > 0
    v_cost, m = vanilla_cost(N, m_0, t)
    v_cost = v_cost * factor
    ineq_cons = {'type': 'ineq', 'fun' : lambda x: v_cost - cost(x, N, m_0)[0]}

    ## use 1 as starting values for cherry-picking
    starting_values = [1 for i in range(t)]
    

    ## call the optimizer
    # maximise the 
    # 5000000 normally
    res = minimize(lambda Kj: m_t(Kj, N, m_0), starting_values, bounds=bound, constraints=ineq_cons,method = "SLSQP", options={'disp': True, "maxiter": maxiter, "eps": eps, "ftol": 1})
    
    ## res.x is the result of the optimization
    final_cost, m_values = cost(res.x, N, m_0) # calculate the final cost and m values
    return res.x, m_0, final_cost, m_values


###############################################################################################################################

# adjust the sumulated cost and m_t
def adjust_simulation(m_values, final_cost, N):
    # try and calculate the last m_t
    m = m_values[-1]
    vcost = final_cost

    # Calculate E1 and E2
    E1 = (1 - 1 / N) ** m
    E2 = (1 - 2 / N) ** m
    average = N * (1 - E1)

    # check variance
    if N * ((N - 1) * E2 + E1 - N * E1 ** 2) < 0:
        # if it is negative (error), set it to 0
        variance = 0
    else:
        # count the variance otherwise
        variance = sqrt(N * ((N - 1) * E2 + E1 - N * E1 ** 2))

    # add on the number of hashes done - no reductions as we are just doing mod which is already fast
    vcost =  vcost + m + 1 / 30 

    # find the m_j+1
    m = average + NormalDist().inv_cdf((1 - pi / 8) / (1 - pi / 4 + 1)) * variance

    # return the adjusted values
    return m, vcost


