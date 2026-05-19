from utils import *
from math import e

N = 2 ** 32
p = 1 - e ** (-2)
alpha = 0.8
t = 5000
cost_factor = 5

result = optimise_kj(N, p, alpha, cost_factor, 1, t=t, maxiter=10000)

# save data in a pkl file
import pickle
with open('N_32_alpha_0.8.pkl', 'wb') as f:
    pickle.dump(result, f)