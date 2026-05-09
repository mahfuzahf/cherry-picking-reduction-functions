# Imports
import cupy as cp
import numpy as np
from numba import cuda
import mmh3
from hashlib import sha256
import random
from time import perf_counter

###############################################################################

# Constants
N = 2 ** 40
alpha = 0.95
t = 10000

# calculate m_0 
def calculate_m0(N, t, alpha):
    mttarget = (2 * N) / (t + 2)
    return (alpha * mttarget) / (1 - alpha)

m0 = calculate_m0(N, t, alpha)

# Hash and reduction functions
def H_c(x):
    return sha256(x.to_bytes(8, 'little')).digest()  # return bytes directly

def r_c(N, t, y, i, ell=0):
    seed = int(i) + (ell*t)
    return mmh3.hash(y, seed, signed=False) % N

###############################################################################




###############################################################################

# build the rainbow table
def build(N, t, m0, kis):

    # IN CPU

    # get parameters for reduction function search
    t_bits = (t-1).bit_length()                     # number of bits to represent t
    k_bits = 32 - t_bits                            # number of bits to represent rf index
    max_k_index = 2**k_bits                         # maximum number of reduction functions per column

    # np array to store fr_indexes
    rf_indexes = np.zeros(t, dtype=np.uint32)

    # initialise the current column with the starting points
    current_column = np.arange(m0, dtype=np.uint64)
    startpoints = np.arange(m0, dtype=np.uint64)

    # for each column
    for i in range(t):

        # get # cherry-picks for this column
        k_i = round(kis[i])

        # get indexes to trial
        pick_sample = np.array(random.sample(range(max_k_index), k_i), dtype=np.uint32)
        index_sample = (i << k_bits) | pick_sample

        # IN GPU

        # move current column to GPU
        current_column_gpu = cp.asarray(current_column)

        # find best seed for this column

        # with the best seed, hash and reduce the current column
        reduced_points = cp.empty(current_column_gpu.size, dtype=cp.uint64)
        # hash and reduce

        # get the indicies of unique points to store the startpoints
        next_column, sp_indicies = cp.unique(reduced_points, return_index=True)
        selected_startpoints = startpoints[sp_indicies]

        # update current column for next iteration
        current_column = next_column.get()  # move back to CPU for next iteration
        startpoints = selected_startpoints.get()  # move back to CPU for next iteration


        # clear GPU memory for next iteration
        del current_column_gpu
        del reduced_points
        del next_column
        del sp_indicies
        del selected_startpoints
        cp.get_default_memory_pool().free_all_blocks()


    return startpoints, current_column
