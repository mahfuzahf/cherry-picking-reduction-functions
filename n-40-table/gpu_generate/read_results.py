import numpy as np

# read results from 216_t80_results.npz and print them
data = np.load("216_t80_results.npz")
startpoints = data['startpoints']
final_column = data['endpoints']
rf_indexes = data['rf_indexes']
print(f"Startpoints: {startpoints}")
print(f"Final column: {final_column}")
print(f"RF indexes: {rf_indexes}")