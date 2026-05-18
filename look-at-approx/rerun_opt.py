# Imports
import pickle
from utils_edited import *
import numpy as np
from math import e
import matplotlib.pyplot as plt
import pandas as pd
import csv


# Constants and Helper Functions
N = 2 ** 24
Ns = [2 ** i for i in range(14, 24, 2)]
t = 80
alphas = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# cost_factor = 1.5
cost_factors = [1.5, 5, 20]
p = 1 - e **(-2)
# n_label = 24
n_labels = [14, 16, 18, 20, 22]

def calculate_m0(N, t, alpha):
    mttarget = (2 * N) / (t + 2)
    return (alpha * mttarget) / (1 - alpha)

def get_hashes(m_values):
    # return sum of all elements in m_vaues apart from last one
    return sum(m_values[:-1])

def get_reductions(kis, m_values):
    reductions = []
    for i in range(len(kis) - 1):
        reduction = m_values[i] * kis[i]
        reductions.append(reduction)
    # return sum of all reductions
    return sum(reductions)


#############################################################################################


for lab, N in enumerate(Ns):
    n_label = n_labels[lab]
    for cost_factor in cost_factors:

        # Get expected vanilla mts
        expected_mts = []
        # mt-max = 2N/t+2
        # mt = alpha * mt-max
        # t = round(log(1-p)/log(1-N**(-1/3)))
        mt_max = (2*N)/(t + 2)
        for alpha in alphas:
            expected_mts.append(round(alpha * mt_max))


        # Optimise
        alpha_kis = []
        alpha_m_vals = []
        alpha_mts = []
        alpha_hashes = []
        alpha_reductions = []
        # for each alpha
        for alpha in alphas:
            if n_label == 22:
                kis, m_0, final_cost, m_values = optimise_kj(N, p, alpha, cost_factor, 1, t, maxiter=100, eps=100)
            else:   
                kis, m_0, final_cost, m_values = optimise_kj(N, p, alpha, cost_factor, 1, t, maxiter=100, eps=1)
            alpha_kis.append(kis)
            alpha_m_vals.append(m_values)
            alpha_mts.append(m_values[-1])
            alpha_hashes.append(get_hashes(m_values))
            alpha_reductions.append(get_reductions(kis, m_values))

            # plot kis 
            plt.plot(range(1, t + 1), kis, label=f'alpha={alpha}')
            plt.xlabel('i') 
            plt.ylabel('k_i') 
            plt.title('Optimal k_i values for different alpha') 
            plt.legend() 
            plt.show() 

        # put into a dataframe
        data_cherry = pd.DataFrame({
            'alpha': alphas,
            'optimiser_mt': alpha_mts,
            'vanilla_expected_mt': expected_mts,
            'hashes': alpha_hashes,
            'reductions': alpha_reductions
        })


        # Plot
        x_indicies = range(len(alphas))
        plt.figure(figsize=(10,6))

        # plot simulated mt values
        plt.plot(x_indicies, data_cherry['optimiser_mt'], color='#648FFF', label='Simulated Cherry $m_t$', linestyle='--', marker='o')

        # plot expected mt values
        plt.plot(x_indicies, expected_mts, color='#FFB000', label='VanillaExpected $m_t$', linestyle='--', marker='o')

        # titles and labels
        plt.xlabel('$\\alpha$ (Maximality Factor)', fontsize=14)
        plt.ylabel('$m_t$', fontsize=14)
        plt.xticks(x_indicies, alphas)
        plt.grid(linestyle="--")
        plt.title(f'Endpoints for Different $\\alpha$ Values ($N=2^{{{n_label}}}$, Cost factor = {cost_factor})', fontsize=16)

        plt.legend()
        plt.tight_layout()  
        # save as eps
        plt.savefig(f'plots/N{n_label}_cf_{cost_factor}.svg', format='svg')
        plt.show()


        # save to a csv file
        data_cherry.to_csv(f'plots/N{n_label}_cf_{cost_factor}.csv', index=False)