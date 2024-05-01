import numpy as np
import matplotlib.pyplot as plt

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize parameters dynamically
        self.weights = np.ones(self.n_components) / self.n_components
        random_row = np.random.randint(low=0, high=n_samples, size=self.n_components)
        self.means = X[random_row]
        self.covariances = np.array([np.eye(n_features)] * self.n_components)


        

        # EM algorithm
        for _ in range(self.max_iter):
            # E-step: Compute responsibilities
            responsibilities = self._compute_responsibilities(X)

            # M-step: Update parameters
            self._update_parameters(X, responsibilities)

            # plot log likelihood
            plt.scatter(_, np.log(np.sum(self._compute_responsibilities(X))))

            # Check convergence

            if np.linalg.norm(responsibilities - self._compute_responsibilities(X)) < self.tol:
                break

    def _compute_responsibilities(self, X):
        n_samples, _ = X.shape
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self._compute_likelihood(X, self.means[k], self.covariances[k])

        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return responsibilities

    # def _compute_likelihood(self, X, mean, covariance):
    #     n_features = X.shape[1]
    #     det = np.linalg.det(covariance)
    #     inv = np.linalg.inv(covariance)
    #     exponent = -0.5 * np.sum((X - mean) @ inv * (X - mean), axis=1)
    #     likelihood = (1.0 / np.sqrt((2 * np.pi) ** n_features * det)) * np.exp(exponent)
    #     return likelihood
    # compute log likelihood using multivariate normal
    

    
    def _update_parameters(self, X, responsibilities):
        n_samples, _ = X.shape
        total_responsibilities = np.sum(responsibilities, axis=0)

        self.weights = total_responsibilities / n_samples
        self.means = (responsibilities.T @ X) / total_responsibilities[:, np.newaxis]
        self.covariances = np.zeros((self.n_components, X.shape[1], X.shape[1]))

        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = (diff.T @ (responsibilities[:, k] * diff)) / total_responsibilities[k]

# Example usage
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
gmm = GaussianMixtureModel(n_components=2)
gmm.fit(X)

# Print results clustered data
responsibilities = gmm._compute_responsibilities(X)
clustered_data = np.argmax(responsibilities, axis=1)
print(clustered_data)
# plot the clustered data

plt.scatter(X[:, 0], X[:, 1], c=clustered_data)
plt.show()

