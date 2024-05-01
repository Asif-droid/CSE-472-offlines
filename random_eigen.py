import numpy as np
# import matplotlib.pyplot as plt
def generate_invertible_matrix(n):
    while True:
        # Create a random n x n matrix with entries
        # random_matrix = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=(n, n))
        random_matrix = np.random.randint(-100, 100, size=(n, n))

        # Check if the matrix is invertible
        if np.linalg.matrix_rank(random_matrix) == n:

            return random_matrix
        

n=input()

A=generate_invertible_matrix(int(n))
print(A)

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigen Value",eigenvalues)
print("Eigen vectors",eigenvectors)

A_reconstructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
print("Reconstructed Matrix",A_reconstructed)
are_matrices_equal = np.allclose(A_reconstructed, A)

if are_matrices_equal:
    print("The matrices are equal.")
else:
    print("The matrices are not equal.")