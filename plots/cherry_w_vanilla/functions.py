from math import pi, sqrt
from statistics import NormalDist
import matplotlib.pyplot as plt

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
