from mle import MLE
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    np.random.seed(42)

    size = 1000
    beta_0 = -2.0
    beta_1 = 0.5

    x = np.random.uniform(1, 12, size)

    sigma_true = 1 / (1 + np.exp(-(beta_0 + beta_1 * x)))
    y = np.random.binomial(1, sigma_true)

    init_beta_0 = 0
    init_beta_1 = 0
    beta_matrix = np.array([init_beta_0, init_beta_1]).reshape((2, 1))
    mlestimator = MLE(beta_matrix=beta_matrix, num_iterations=100)
    mlestimator.fit(x, y)

    x_axis = np.arange(-2, 15)
    y_true = 1 / (1 + np.exp(-(beta_0 + beta_1 * x_axis)))
    y_hat = 1 / (1 + np.exp(-(mlestimator.beta_matrix[0].item() + mlestimator.beta_matrix[1].item() * x_axis)))
    plt.figure()
    plt.plot(x_axis, y_true)
    plt.plot(x_axis, y_hat)
    plt.legend(
        [
            r"True curve $\frac{1}{1 + e^-(\beta_0 + \beta_1 * x)}$ = $\frac{1}{1 + e^-(-2.0 + 0.5 * x)}$",
            r"Hat curve $\frac{1}{1 + e^-(\hat{\beta_0} + \hat{\beta_1} * x)}$ = $\frac{1}{1 + e^-(-1.94 + 0.48 * x)}$",
        ]
    )
    plt.xlabel("Child age")
    plt.ylabel("Probability")
    plt.grid()
    plt.show()
