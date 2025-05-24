from dataclasses import dataclass
from tqdm import tqdm
import numpy as np


@dataclass
class MLE:
    beta_matrix: np.ndarray
    num_iterations: int

    def get_sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def get_gradiant_matrix(self, x_matrix, y_label):
        z = x_matrix @ self.beta_matrix
        sigma = self.get_sigmoid(z)

        error = y_label.reshape((-1, 1)) - sigma
        grad_matrix = x_matrix.T @ error

        return grad_matrix

    def get_hessian_matrix(self, x_feature):
        z = self.beta_matrix[0] + self.beta_matrix[1] * x_feature
        sigma = self.get_sigmoid(z)
        s = sigma * (1 - sigma)

        dl2_dw02 = -np.sum(s)
        dl2_dw0w1 = -np.sum(s * x_feature)
        dl2_dw12 = -np.sum(s * (x_feature**2))

        H_matrix = np.array([[dl2_dw02, dl2_dw0w1], [dl2_dw0w1, dl2_dw12]])

        return H_matrix

    def update_features(self, x_matrix, y_label):
        grad_matrix = self.get_gradiant_matrix(x_matrix, y_label)
        H_matrix = np.linalg.inv(self.get_hessian_matrix(x_matrix[:, 1]))
        self.beta_matrix = self.beta_matrix - (H_matrix @ grad_matrix)

    def fit(self, x_feature: np.ndarray, y_label: np.ndarray):
        x_matrix = np.array([np.ones_like(x_feature), x_feature]).T
        for idx in tqdm(range(self.num_iterations)):
            self.update_features(x_matrix, y_label)

        self.beta_0 = self.beta_matrix[0]
        self.beta_1 = self.beta_matrix[1]
        print(f"Beta_0: {self.beta_0.item()}")
        print(f"Beta_1: {self.beta_1.item()}")
