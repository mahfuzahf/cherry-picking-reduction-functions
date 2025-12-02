# As we have the same number of K_i's as columns, we need to use the last K_i to figure out the true m_t, as that is what we do when building it

# imports
from math import sqrt, pi, log, e
import pickle
from statistics import NormalDist
import pandas as pd
import functions as func

# constants
N = 2 ** 16

# help us calculate the true m_t and cost to add on
def calc_true_mt_and_cost(m, k_t, cost):

    original_m = m  # store original m value for cost calculation later
    actual_cost = cost

    # this is doing some sort of stats to find the expected m_i+1 value for the column 
    # for k in k_t:  # iterate through the column
    # Calculate E1 and E2
    E1 = (1 - 1 / N) ** m
    E2 = (1 - 2 / N) ** m
    # calculate the statistical average of m_j+1
    average = N * (1 - E1)
    # check if the variance is positive, sometimes error happens because of float calculation on very small values
    if N * ((N - 1) * E2 + E1 - N * E1 ** 2) < 0:
        # if it is negative (error), set it to 0
        variance = 0
    else:
        # count the statistical variance of m_j+1 otherwise
        variance = sqrt(N * ((N - 1) * E2 + E1 - N * E1 ** 2))

    # add to actual cost
    actual_cost += original_m + original_m * k_t / 577.44
    # find the m_j+1
    m = average + NormalDist().inv_cdf((k_t - pi / 8) / (k_t - pi / 4 + 1)) * variance

    return m, actual_cost


def replace_data():

    # place to store costs and simulated m_t values
    m_ts = []
    costs = []


    # load in all the tables
    cost_factors = range(5, 71, 5) # cost factors from 5 to 70

    # load in pickle file with current sim results
    with open("higher_costs_ftol_1.pickle", "rb") as f:
        data = pickle.load(f)

    # structure of file
    # array of different alpha used
        # for each alpha, array of different costs used
        # for each cost, array arranged as [Kjs, m_0, final_cost, m_values]

    for i in range(len(cost_factors)):
        # run function to calculate true m_t and cost
        true_mt, total_cost = calc_true_mt_and_cost(data[-1][i][3][-1], data[-1][i][0][-1], data[-1][i][2])
        # store values
        m_ts.append(round(true_mt))
        costs.append(total_cost)

    print(m_ts)
    print(costs)


    # now replace simulated_mt and simulated_cost in csv file
    df = pd.read_csv("results/building/building_table_results.csv")

    df["simulated_mt"] = m_ts
    df["simulated_cost"] = costs

    df.to_csv("results/building/building_table_results_updated.csv", index=False)


# # I want to see what the startpoints for the simulations were
# # open the pickle file
# with open("higher_costs_ftol_1.pickle", "rb") as f:
#     data = pickle.load(f)

# # get the alpha = 0.95 for different cost factors 
# # array of different alpha used
#     # for each alpha, array of different costs used
#     # for each cost, array arranged as [Kjs, m_0, final_cost, m_values]

# for cost_factor_data in data[-1]:
#     m = cost_factor_data[3][-1] # m_value for last column
#     k = cost_factor_data[0][-1] # k_value for last column
#     print("m_t-1:", m)

#     # Calculate E1 and E2
#     E1 = (1 - 1 / N) ** m
#     E2 = (1 - 2 / N) ** m
#     # calculate the statistical average of m_j+1
#     average = N * (1 - E1)
#     print("average:", average)
#     # check if the variance is positive, sometimes error happens because of float calculation on very small values
#     if N * ((N - 1) * E2 + E1 - N * E1 ** 2) < 0:   # this equation gives the variance, so sqrt to get starndard deviation
#         # if it is negative (error), set it to 0
#         sd = 0
#         print("sd: 0")
#     else:
#         # count the statistical variance of m_j+1 otherwise
#         sd = sqrt(N * ((N - 1) * E2 + E1 - N * E1 ** 2))
#         print("sd:", sd)
#     # find the m_j+1
#     m = average + NormalDist().inv_cdf((k - pi / 8) / (k - pi / 4 + 1)) * sd
#     print("calculated m_t:", m)


# build a vanilla table with N = 2^42, alpha = 0.5 
N = 2 ** 42
p = 1-e**-2 # our table coverage
t = round(log(1-p)/log(1-N**(-1/3))) # Calculate t

# 

table = func.build_table(N, t, )
