# Imports
from math import e
from hashlib import sha256
import mmh3
from radix import radix_sort
import numpy as np
from time import perf_counter
import pandas as pd


# Constants
N = 2 ** 16  # keyspace
p = 1 - e ** -2  # our table coverage - 86%
t = 80  # number of columns
k = 100 # number of trials


#######################################################################################################################


## Hash functions
def H(x):
    return int.from_bytes(sha256(x.to_bytes(8)).digest())

def H_c(x):
    return sha256(x.to_bytes(8)).digest()  # return bytes directly


#######################################################################################################################


## Reduction functions

# regular mod reduction function
def r_mod(N, t, y, i, ell=0):   # also takes in ell - number of tables - for future use (but currently ell=0)
	# return (y + i + ell*t) % N
    # if everything is bytes

    return (y + i + ell*t) & (N - 1) # N is a power of 2 so can use bitwise AND instead of modulo
    
# mmh reduction function
def r_mmh(N, t, y, i, ell=0):
    return mmh3.hash(y, i + (ell*t), signed=False) % N


#######################################################################################################################

# create dataframe to store results
results_df = pd.DataFrame(columns=['Trial', 'Hash Time (s)'])


# create a numpy array of 10^7 numbers
S = np.arange(10**7)

# hash the numbers using H_c - 10 trials
for i in range(10):
	start_time = perf_counter()
	S_hashed = np.array([H_c(x) for x in S])	
	duration = perf_counter() - start_time
	# append the result to the dataframe
	results_df = pd.concat([results_df, pd.DataFrame({'Trial': [i +1], 'Hash Time (s)': [duration]})], ignore_index=True)
     
# display mean hash time 
print("Mean Hash Time (s):", results_df['Hash Time (s)'].mean())

# # k rounds of reduction and deduplication
# for i in range(k):
#     S_reduced = np.array([r_mod(N, t, x, i) for x in S_hashed])
#     S_sorted = radix_sort(S_reduced)