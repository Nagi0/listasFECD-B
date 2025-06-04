from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_expected_value(p_params: list, p_interval: tuple, p_size: int):
    stage = np.linspace(p_interval[0], p_interval[1], p_size)
    gender_female = np.zeros(p_size)
    gender_male = np.ones(p_size)
    female = (np.exp(-p_params[0])) * ((np.exp(-p_params[1])) ** gender_female) * ((np.exp(-p_params[2])) ** stage)
    male = (np.exp(-p_params[0])) * ((np.exp(-p_params[1])) ** gender_male) * ((np.exp(-p_params[2])) ** stage)

    plt.figure()
    plt.title("Comparação do tempo de vida esperado")
    plt.plot(stage, male, label="Homens")
    plt.plot(stage, female, label="Mulheres")
    plt.gca().invert_xaxis()
    plt.xlabel("Estágio da doença")
    plt.ylabel("Tempo de vida esperado")
    plt.legend()
    plt.grid()
    plt.show()


@dataclass
class ExpMLE:
    params: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    lambda_array: np.ndarray

    def check_converged(self, p_grad_matrix):
        return not np.any(np.round(p_grad_matrix, 7))

    def update_lambda(self):
        self.lambda_array = np.exp(self.X @ self.params)

    def update_params(self, p_hessian_matrix, p_grad_matrix):
        self.params = self.params - (np.linalg.inv(p_hessian_matrix) @ p_grad_matrix)

    def get_hessian_matrix(self):
        return -self.X.T @ ((self.Y * self.lambda_array) * self.X)

    def get_grad_matrix(self):
        return self.X.T @ (1 - self.Y * self.lambda_array)

    def newton_raphson(self, max_iter=100):
        for iter in tqdm(range(max_iter)):
            grad_matrix = self.get_grad_matrix()
            hessian_matrix = self.get_hessian_matrix()

            self.update_params(hessian_matrix, grad_matrix)
            self.update_lambda()

            if self.check_converged(grad_matrix):
                break

        print(f"Theta matrix: {self.params.T}")
