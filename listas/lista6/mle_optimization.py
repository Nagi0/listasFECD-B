from dataclasses import dataclass
from typing import List
from math import log, sqrt, pi
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


@dataclass
class MLEGaussian:
    sigma2: float
    mu_true: float
    sample: list[float]

    def get_log_likelihood(self, interval: tuple, size: int, plot: bool = True):
        interval_values = np.linspace(interval[0], interval[1], size)

        log_lik_list = []
        for mu_value in interval_values:
            log_lik = -log(sqrt(2 * pi * self.sigma2)) - (1 / (2 * len(self.sample))) * (
                np.sum(((self.sample - mu_value) / sqrt(self.sigma2)) ** 2)
            )
            log_lik_list.append(log_lik)

        if plot:
            plt.figure()
            plt.plot(interval_values, log_lik_list)
            plt.grid()
            plt.show()

        return log_lik_list

    def get_score_function(self, interval: tuple, size: int, plot: bool = False):
        interval_values = np.linspace(interval[0], interval[1], size)
        score_function = (1 / self.sigma2) * (np.sum(self.sample) - len(self.sample) * interval_values)
        mu_optimal = np.sum(self.sample) / len(self.sample)

        if plot:
            plt.figure()
            plt.plot(interval_values, score_function)
            plt.axvline(x=self.mu_true, color="green")
            plt.axvline(x=mu_optimal, color="red")
            plt.grid()
            plt.legend(["Score Function", "True Value", r"$\hat{\mu}(y)$"])
            plt.show()

        return score_function, mu_optimal


@dataclass
class MLEPoisson:
    lambda_true: float
    sample: list[float]

    def dl_dlambda(self, param):
        return (np.sum(self.sample) / param) - (len(self.sample) / (1 - np.exp(-param)))

    def dl_dlambda_prime(self, param):
        return -(np.sum(self.sample) / param**2) - (len(self.sample) * np.exp(-param)) / (1 - np.exp(-param)) ** 2

    def get_score_function(self, interval: tuple, size: int, plot: bool = False):
        interval_values = np.linspace(interval[0], interval[1], size)
        score_function = (np.sum(self.sample) / interval_values) - (len(self.sample) / (1 - np.exp(-interval_values)))

        lambda_optimal = root_scalar(self.dl_dlambda, fprime=self.dl_dlambda_prime, x0=self.sample[0], method="newton")
        print(lambda_optimal)

        if plot:
            plt.figure()
            plt.plot(interval_values, score_function)
            plt.axvline(x=self.lambda_true, color="green")
            plt.axvline(x=lambda_optimal.root, color="red")
            plt.grid()
            plt.legend(["Score Function", "True Value", r"$\hat{\lambda}(y)$"])
            plt.show()

        return score_function, lambda_optimal


def compare_optmizers(optimizers: List[MLEGaussian], interval: tuple, size: int, mu_true: float):
    score_function_list = []
    mu_hat_list = []

    for opt in optimizers:
        score_values, mu_hat = opt.get_score_function(interval, size)
        score_function_list.append(score_values)
        mu_hat_list.append(mu_hat)

    plt.figure()
    interval_values = np.linspace(interval[0], interval[1], size)
    for idx, (sc_f, mu_h) in enumerate(zip(score_function_list, mu_hat_list)):
        plt.plot(interval_values, sc_f, color=f"C{idx}", label=f"Sample {idx+1}")
        plt.axvline(x=mu_h, ls="--", color=f"C{idx}", label=r"$\hat{\mu}" + f"{idx+1}" + r"$")
    plt.axvline(x=mu_true, color="red", label="True Value")
    plt.legend()
    plt.grid()
    plt.show()


def compare_log_lik(optimizers: List[MLEGaussian], interval: tuple, size: int, mu_true: float, sigma2_true: float):
    log_lik_list = []

    for opt in optimizers:
        log_lik_values = opt.get_log_likelihood(interval, size, plot=False)
        log_lik_list.append(log_lik_values)

    plt.figure()
    interval_values = np.linspace(interval[0], interval[1], size)
    for idx, lg_lk in enumerate(log_lik_list):
        plt.plot(interval_values, lg_lk, color=f"C{idx}", label=f"Sample {idx+1}")

    mu_true_list = []
    for mu_val in interval_values:
        mu_true_value = -log(sqrt(2 * pi * sigma2_true)) - (1 / (2 * sigma2_true)) * (
            sigma2_true + (mu_val - mu_true) ** 2
        )
        mu_true_list.append(mu_true_value)

    plt.plot(interval_values, mu_true_list, color="red", label="Expected Log-Lik")
    plt.legend()
    plt.title(f"Log-Likelihood for samples with n={len(optimizers[0].sample)}")
    plt.grid()
    plt.show()
