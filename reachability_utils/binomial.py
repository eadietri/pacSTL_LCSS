import math
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from ellipsoids import in_ellipsoid
from scipy.stats import binom, beta



# Calculate misses for binomial tail inversion for ellipsoidal tube:
def calculate_epsilon_misses(data, A_star, b_star, num_samples):

    misses = 0
    for p in data:
        test = in_ellipsoid(A_star, b_star, p)
        if test - 1 > 0:
            misses += 1
    return misses

def calculate_epsilon_tube_ellipsoid(misses, num_samples):
    # The optimal value of p
    p_estimate = binomial_tail(misses, num_samples)
    return p_estimate

# Calculate misses for binomial tail inversion for ellipsoid:
def calculate_epsilon_ellipsoid(data, A_star, b_star, num_samples):

    misses = 0
    for i in range(len(data)):
        test = in_ellipsoid(A_star, b_star, data[i])
        if test - 1 > 0:
            misses += 1

    # The optimal value of p
    p_estimate = binomial_tail(misses, num_samples)
    return p_estimate

# Binomial tail inversion to calculate accuracy level given a test datasest
def binomial_tail(misses: int, num_samples: int) -> float:
    """
    Compute the largest p such that:
        BinomialCDF(k=misses; n=num_samples, p) >= 1e-9
    """
    k = misses
    n = num_samples

    def objective(p):
        return -p  # maximize p
    
    def binom_cdf(e, k, l):
        return binom.cdf(k, l, e)
    
    def binom_cdf_if_fails(e, k, l):
        binom_sum = 0
        for j in range(0, k+1):
            if not math.isnan(binom.pmf(j, l, e)[0]):
                binom_sum += binom.pmf(j, l, e)
        return binom_sum
    
    lc = NonlinearConstraint((lambda p : binom_cdf(p, k, n) - 0.000000001), lb=0, ub=1)
    result = minimize(objective, x0=k/n, constraints=lc, method='SLSQP')

    if not result.success:
        print("Optimization failed:", result.message, " re-running with different cdf function")
        lc = NonlinearConstraint((lambda p : binom_cdf_if_fails(p, k, n) - 0.000000001), lb=0, ub=1)
        result = minimize(objective, x0=k/n, constraints=lc, method='SLSQP')
        if not result.success:
            print("Optimization failed again:", result.message, "use beta.ppf:")
            return 1- beta.ppf(0.000000001, num_samples-misses, misses+1)

    return result.x[0]