from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class MLEGaussian:
    sigma2: float
    mu_true: float
    sample: list[float]

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
    lambda_hat: float
    lambda_true: float
    sample: list[float]


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
