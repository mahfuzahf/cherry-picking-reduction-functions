from utils import *
import pickle

with open("cherry_N_16_different_alpha_0.5_0.95_optimisation_results.pkl", "rb") as f:
    data = pickle.load(f)

for item in data:
    # print m_t for each alpha
    print(item[3][-1])

