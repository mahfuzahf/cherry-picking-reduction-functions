#Imports
import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import sqrt, pi, log
from statistics import NormalDist


# Constants
N = 20000
alpha = 0.8
t=100

#Functions 

def calculate_m0(N, t, alpha):
    mttarget = (2 * N) / (t + 2)
    return (alpha * mttarget) / (1 - alpha)

def dp_matrix(N, m):

    """
    Returns an array of probabilites for each index number of bins to be occupied by m balls
    """

    # only store one array instead of a matrix, since we only need the previous row to calculate the current row
    prev = np.zeros(N + 1, dtype=np.float64)
    prev[0] = 1.0

    for i in range(1, m + 1):
        # next row
        curr = np.zeros(N + 1, dtype=np.float64)
        # for all bins
        j = np.arange(min(i, N) + 1)

        # if ball lands in a new bin (j > 0)
        # create an array of which elements of j are above 0
        # divide by N to get probabilities
        mask_new = j > 0
        curr[j[mask_new]] += prev[j[mask_new] - 1] * (N - (j[mask_new] - 1)) / N

        # if ball lands in existing bin (j <= i - 1)
        # create an array of which elements of j are less than or equal to i - 1
        # divide by N to get probabilities
        mask_existing = j <= i - 1
        curr[j[mask_existing]] += prev[j[mask_existing]] * j[mask_existing] / N

        # set prev to curr for next iteration
        prev = curr

    return prev[:min(m, N) + 1]


def dp_occupancy(N, m, j):
    matrix = dp_matrix(N, m)
    return matrix[j] if j < len(matrix) else 0.0


def dp_calc_max(N, m, k):
    matrix = dp_matrix(N, m)

    # Calculate CDFs using cumulative sum
    cdfs = np.cumsum(matrix)
    
    # Raise to power k
    cdf_powers = cdfs**k
    
    # Sum (1 - cdf^k) for y=0 to m
    return np.sum(1 - cdf_powers)


def get_mean(N, m):
    return N - (N*(1 - 1/N)**m)

def get_std(N, m):
    # E1 and E2
    E1 = (1 - 1 / N) ** m
    E2 = (1 - 2 / N) ** m

    return sqrt(N * (((N - 1) * E2) + E1 - (N * (E1 ** 2))))

def approx_mi(N, m, k):
    average = get_mean(N, m)
    sd = get_std(N, m)
    m = average + NormalDist().inv_cdf((k - pi / 8) / (k - pi / 4 + 1)) * sd
    return m


# Table with the classical occupancy problem
def table_cop(N, t, alpha, k_values):
    m0 = round(calculate_m0(N, t, alpha))
    m_values = [m0]
    m = m0

    for i in range(t):
        m = round(dp_calc_max(N, m, k_values[i]))
        m_values.append(m)

    return m_values


# Table with the approximation of the occupancy problem
def table_approx(N, t, alpha, k_values):
    m0 = round(calculate_m0(N, t, alpha))
    m_values = [m0]
    m = m0

    for i in range(t):
        m = approx_mi(N, m, k_values[i])
        m_values.append(m)

    return m_values


if __name__ == "__main__":
    # generate each table 10 times

    # load k values from file
    k_values = np.load("cherry-picking-reduction-functions/approx/cherry_k_values.npy")

    # make results directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # generate table for cop


    for i in range(10):
        m_values_cop = table_cop(N, t, alpha, k_values)
        np.save(os.path.join(output_dir, f"{output_dir}/cop_table_{i}.npy"), m_values_cop)

        m_values_approx = table_approx(N, t, alpha, k_values)
        np.save(os.path.join(output_dir, f"{output_dir}/approx_table_{i}.npy"), m_values_approx)

        