import numpy as np
from mle_optimization import *

if __name__ == "__main__":
    mle1 = MLEGaussian(4.0, 10.0, [10.82, 9.76, 10.55, 7.09, 10.84])
    mle1.get_score_function((8, 12), 10)

    mle2 = MLEGaussian(4.0, 10.0, [12.98, 10.73, 8.49, 8.45, 9.78])
    mle2.get_score_function((8, 12), 10)

    compare_optmizers([mle1, mle2], (8, 12), 10, 10.0)

    mle_poisson = MLEPoisson(1.2, [1, 2, 1, 3, 2])
    mle_poisson.get_score_function((1.0, 1.6), 10, plot=True)

    mle1 = MLEGaussian(1.7**2, 10.0, [7.10, 7.09, 7.12, 12.62, 10.03])
    mle1.get_log_likelihood((7, 13), 50, plot=False)

    mle2 = MLEGaussian(1.7**2, 10.0, [9.45, 11.62, 9.46, 12.47, 12.05])
    mle2.get_log_likelihood((7, 13), 50, plot=False)

    compare_log_lik([mle1, mle2], (7, 13), 50, 10, 1.7**2)

    mle1 = MLEGaussian(1.7**2, 10.0, list(np.random.normal(10, 1.7, 50)))
    mle1.get_log_likelihood((7, 13), 50, plot=False)

    mle2 = MLEGaussian(1.7**2, 10.0, list(np.random.normal(10, 1.7, 50)))
    mle2.get_log_likelihood((7, 13), 50, plot=False)

    compare_log_lik([mle1, mle2], (7, 13), 50, 10, 1.7**2)
