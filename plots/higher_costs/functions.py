from math import pi, sqrt
from statistics import NormalDist
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# constants for plotting
NUM_COLORS = 20
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)

## define the objective and target functions
def cost(inputs, N, m_0):  ## the cost calculation function as objective function to be minimized
    ### the input variables to find are as follows:
    # input[0] is the number of starting points
    # input[1:] is the K_j for all 1<=j<=t
    K_js = inputs[1:]  # K_j values
    cost = 0  # cost counter
    m_values = [m_0]  # list to keep the mj values

    for k in K_js:  # iterate through the column
        # Calculate E1 and E2
        E1 = (1 - 1 / N) ** m_0
        E2 = (1 - 2 / N) ** m_0
        # calculate the statistical average of m_j+1
        average = N * (1 - E1)
        ## check if the variance is positive, sometimes error happens because of float calculation on very small values
        if N * ((N - 1) * E2 + E1 - N * E1 ** 2) < 0:
            # if it is negative (error), set it to 0
            variance = 0
        else:
            # count the variance otherwise
            variance = sqrt(N * ((N - 1) * E2 + E1 - N * E1 ** 2))
        # count the cost before defining the new m
        cost += m_0 + m_0 * k / 577.44
        # find the m_j+1
        m_0 = average + NormalDist().inv_cdf((k - pi / 8) / (k - pi / 4 + 1)) * variance
        m_values.append(m_0)
        # count the cost

    # return cost - not square rooted as we are bounding cost
    return cost, m_values


# %%
def m_t(inputs, N, m_0):  ## the m_t function as a target function
    ### the input variables to find are as follows:
    # input[0] is the number of starting points
    # input[1:] is the K_j for all 1<=j<=t

    K_js = inputs[1:]  # K_j values
    m = m_0  # starting m value

    for k in K_js:  # iterate through the column
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
        # find the m_j+1
        m = average + NormalDist().inv_cdf((k - pi / 8) / (k - pi / 4 + 1)) * variance

    # return -m to maximise the amount of points with optimiser
    return -m


# %%
# function to get cost of vanilla rainbow table
def vanilla_cost(N, m_0, t):
    # keep track of cost
    cost = 0
    m = m_0
    m_values = [m]

    # iterate through each column
    for i in range(t):
        # Calculate E1 and E2
        E1 = (1 - 1 / N) ** m
        E2 = (1 - 2 / N) ** m
        average = N * (1 - E1)

        # check variance
        if N * ((N - 1) * E2 + E1 - N * E1 ** 2) < 0:
            # if it is negative (error), set it to 0
            variance = 0
        else:
            # count the variance otherwise
            variance = sqrt(N * ((N - 1) * E2 + E1 - N * E1 ** 2))

        # calculate cost - add 1 for each hash we do (m hashes in this column)
        cost = cost + m

        # find the m_j+1
        m = average + NormalDist().inv_cdf((1 - pi / 8) / (1 - pi / 4 + 1)) * variance
        m_values.append(m)

    return cost, m_values


def plot_kjs(K_js, m_0, alpha, N):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(K_js, "k", label="K")
    ax.set_ylabel(r"$K_{j}$", **{"fontname": "Times New Roman", "style": "italic", "fontsize": 12, "labelpad": -2})
    ax.set_xlabel("j", **{"fontname": "Times New Roman", "style": "italic", "fontsize": 12})

    fig.suptitle(f"K curve where $m_0$ = {m_0} ($α$ = {alpha}), N = {N}, t = {len(K_js)}")


def plot_kjs_cost(K_js, m_0, alpha, factor):
    # make filepath
    filepath = generate_filepath(alpha, factor)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(K_js, "k", label="K")
    ax.set_ylabel(r"$K_{j}$", **{"fontname": "Times New Roman", "style": "italic", "fontsize": 12, "labelpad": -2})
    ax.set_xlabel("j", **{"fontname": "Times New Roman", "style": "italic", "fontsize": 12})

    fig.suptitle(f"K curve where $α$ = {alpha}, bounded by {factor}P(V) - P(C) > 0 ")
    plt.savefig(filepath)
    plt.close(fig)


def plot_all_kj(results, labels, legend_label, alpha):
    # get filepath
    filepath = generate_filepath_all_alpha(alpha)
    print(filepath)

    # get plot title
    title = generate_title_all_alpha(alpha)

    # to get different colours
    sns.reset_orig()  # get default matplotlib styles back
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS)  # a list of RGB tuples

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # [K_js, m_0, final_cost, m_values]
    for i, result in enumerate(results):
        K_js = result[0]
        label = labels[i]
        lines = ax.plot(K_js, label=f"{legend_label} = {label}")
        lines[0].set_color(clrs[i])
        lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])

    ax.set_ylabel(r"$K_{j}$", **{"fontname": "Times New Roman", "style": "italic", "fontsize": 12, "labelpad": -2})
    ax.set_xlabel("j", **{"fontname": "Times New Roman", "style": "italic", "fontsize": 12})

    fig.suptitle(title)
    # Position legend outside the plot (right side)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # Adjust layout to fit the legend outside
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)


def generate_filepath(alpha, factor):
    return f"{alpha}_different_costs/K_curve_{alpha}_{factor}.png"


def generate_filepath_all_alpha(alpha):
    return f"{alpha}_different_costs/K_curve_all_alpha_{alpha}_new.png"


def generate_title_all_alpha(alpha):
    return f"K curve where $α$ = {alpha} for different costs"


## put all the plots on one figure
def draw_subplot(ax, results, labels, legend_label, alpha, clrs):
    # get plot title
    title = generate_title_all_alpha(alpha)

    # [K_js, m_0, final_cost, m_values]
    for i, result in enumerate(results):
        K_js = result[0]
        label = labels[i]
        lines = ax.plot(K_js, label=f"{legend_label} = {label}")
        lines[0].set_color(clrs[i])
        lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])

    ax.set_title(title)
    ax.set_ylabel(r"$K_{j}$", **{"fontname": "Times New Roman", "style": "italic", "fontsize": 12, "labelpad": -2})
    ax.set_xlabel("j", **{"fontname": "Times New Roman", "style": "italic", "fontsize": 12})


def plot_all(results, labels, legend_label, alphas):

    # set up the figure with subplots
    rows, cols = 2, 4  # Define a 2-row, 4-column layout
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(24 , 8), sharex=True)

    # to get different colours
    sns.reset_orig()  # get default matplotlib styles back
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS)  # a list of RGB tuples

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i, result in enumerate(results):
        ax = axes[i]
        draw_subplot(ax, result, labels, legend_label, alphas[i], clrs)

    # Create a single legend from the first subplot
    legend_lines = [plt.Line2D([0], [0], color=clrs[i], linestyle=LINE_STYLES[i % NUM_STYLES]) for i in range(20)]

    fig.legend(legend_lines, labels, loc="upper right", bbox_to_anchor=(1.2, 1), title=legend_label)

    # Set common x-label
    axes[-1].set_xlabel("j", fontname="Times New Roman", style="italic", fontsize=12)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for the legend

    plt.show()

