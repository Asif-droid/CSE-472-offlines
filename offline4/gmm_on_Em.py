import numpy as np

class GMM:
    def __init__(self, n_components):
        self.n_components = n_components
        self.weights = None
        self.means = None
        self.covariances = None

    def fit(self, X, max_iterations=100, tol=1e-4):
        n_samples, n_features = X.shape

        # Initialize model parameters
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features)] * self.n_components)

        prev_log_likelihood = None
        for iteration in range(max_iterations):
            # E-step: Compute responsibilities
            responsibilities = self._expectation(X)

            # M-step: Update model parameters
            self._maximization(X, responsibilities)

            # Compute log likelihood
            log_likelihood = self._compute_log_likelihood(X)

            # Check for convergence
            if prev_log_likelihood is not None and log_likelihood - prev_log_likelihood < tol:
                break

            prev_log_likelihood = log_likelihood

    def _expectation(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for i in range(self.n_components):
            mean = self.means[i]
            covariance = self.covariances[i]
            weight = self.weights[i]

            # Compute the probability density function of each sample
            pdf = self._multivariate_normal_pdf(X, mean, covariance)

            # Compute the responsibility of each component for each sample
            responsibilities[:, i] = weight * pdf

        # Normalize the responsibilities
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        return responsibilities

    def _maximization(self, X, responsibilities):
        n_samples = X.shape[0]

        # Update the weights
        self.weights = np.mean(responsibilities, axis=0)

        # Update the means
        weighted_sum = np.dot(responsibilities.T, X)
        self.means = weighted_sum / np.sum(responsibilities, axis=0, keepdims=True)

        # Update the covariances
        for i in range(self.n_components):
            mean = self.means[i]
            diff = X - mean
            weighted_diff = responsibilities[:, i].reshape(-1, 1) * diff
            covariance = np.dot(weighted_diff.T, diff)
            self.covariances[i] = covariance / np.sum(responsibilities[:, i])

    def _multivariate_normal_pdf(self, X, mean, covariance):
        n_features = X.shape[1]
        det = np.linalg.det(covariance)
        inv = np.linalg.inv(covariance)
        norm_const = 1.0 / ((2 * np.pi) ** (n_features / 2) * np.sqrt(det))
        exponent = -0.5 * np.sum(np.dot((X - mean), inv) * (X - mean), axis=1)
        return norm_const * np.exp(exponent)

    def _compute_log_likelihood(self, X):
        log_likelihood = 0

        for i in range(self.n_components):
            mean = self.means[i]
            covariance = self.covariances[i]
            weight = self.weights[i]

            # Compute the probability density function of each sample
            pdf = self._multivariate_normal_pdf(X, mean, covariance)

            # Compute the log likelihood
            log_likelihood += np.sum(np.log(weight * pdf))

        return log_likelihood
