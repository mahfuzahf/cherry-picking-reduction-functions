import pickle

with open(r"C:\Users\Mahfuzah Fariha\PhD Repositories\choice-of-reduction-functions\cherry-picking-reduction-functions\plots\higher_costs\N_16\constructing_table\simulations\cherry_N_20_different_alpha_0.5_0.95_optimisation_results.pkl", "rb") as f:
    up = pickle.Unpickler(f)
    up.persistent_load = lambda x: None  # avoid loading full objects
    try:
        obj = up.load()
    except Exception as e:
        print("Partial load error:", e)
