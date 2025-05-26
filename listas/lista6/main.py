from mle_optimization import *

if __name__ == "__main__":
    mle1 = MLEGaussian(4.0, 10.0, [10.82, 9.76, 10.55, 7.09, 10.84])
    mle1.get_score_function((8, 12), 10, plot=True)

    mle2 = MLEGaussian(4.0, 10.0, [12.98, 10.73, 8.49, 8.45, 9.78])
    mle2.get_score_function((8, 12), 10)

    compare_optmizers([mle1, mle2], (8, 12), 10, 10.0)
