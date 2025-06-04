from exp_mle import ExpMLE, plot_expected_value
import numpy as np


if __name__ == "__main__":
    y = np.array([3.19, 16.87, 24.65, 2.04, 5.73, 1.03, 6.02, 42.41, 36.08, 7.34, 24.88, 5.90])[:, np.newaxis]
    x1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    x2 = [11, 67, 92, 32, 85, 36, 20, 69, 58, 47, 100, 72]
    x = np.stack([np.ones_like(x1), x1, x2], axis=1)

    plot_expected_value([-2.0, -0.4, -0.005], (100, 10), 10)

    lambda_array = (1 / np.mean(y)) * np.ones_like(y)
    init_params = np.array([-np.log(np.mean(y)), 0, 0])[:, np.newaxis]

    mle = ExpMLE(init_params, x, y, lambda_array)
    mle.newton_raphson(max_iter=10)
