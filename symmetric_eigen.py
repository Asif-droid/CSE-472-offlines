import numpy as np

def generate_invertible_symmetric_matrix(n):
    
    while True:
       # Generate a random lower triangular matrix with integers
        lower_triangular = np.tril(np.random.randint(-100, 100, size=(n, n)), k=-1)

        # Create a symmetric matrix by filling the upper triangular part
        symmetric_matrix = lower_triangular + lower_triangular.T

        # Set the diagonal to random integers
        np.fill_diagonal(symmetric_matrix, np.random.randint(-100, 100, size=n))

        # Check if the matrix is invertible by measuring rank
        if np.linalg.matrix_rank(symmetric_matrix) == n:
            return symmetric_matrix.astype(int)


n2=input()

As=generate_invertible_symmetric_matrix(int(n2))
print(As)

eigenvalues, eigenvectors = np.linalg.eig(As)

print("Eigen value ",eigenvalues)
print("Eigen vectors ",eigenvectors)


As_reconstructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)

print("Reconstructed Matrix",As_reconstructed)

are_matrices_equal = np.allclose(As_reconstructed, As)

if are_matrices_equal:
    print("The matrices are  equal.")
else:
    print("The matrices are not equal.")
