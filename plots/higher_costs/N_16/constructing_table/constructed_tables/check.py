# count how many unique reduction function indexes each table has

# imports
import pickle

cost_factors = list(range(5, 101, 5))

for item in cost_factors:
    with open(f'cherry_indexes_alpha_0.95_t_80_cost_{item}.pkl', 'rb') as f:
        indexes = pickle.load(f)
        f.close()
    print(len(set(indexes)))
    # print(indexes)