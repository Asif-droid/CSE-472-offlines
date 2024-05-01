# -*- coding: utf-8 -*-
"""offline4.ipynb


Original file is located at
    https://colab.research.google.com/drive/1X1aq8A6SRdZMoVOC7xMYYTHERroQ6U4G
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

import imageio

def read_data_6d(file_path):
  df_6d = pd.read_csv(file_path, sep=',', header=None, names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])

  # Display the DataFrame
  # print(df_6d)
  return df_6d

def read_data_3d(file_path):
  df_6d = pd.read_csv(file_path, sep=',', header=None, names=['col1', 'col2', 'col3'])

  # Display the DataFrame
  # print(df_6d)
  return df_6d

def read_data_2d(file_path):
  df_6d = pd.read_csv(file_path, sep=',', header=None, names=['col1', 'col2'])

  # Display the DataFrame
  # print(df_6d)
  return df_6d

def apply_pca(df):
  cols=df.shape[1]
  pc_df=None

  if(cols>2):

    X = df.iloc[:, :6]

    # Standardize the data (optional but often recommended for PCA)
    X_standardized = (X - X.mean()) / X.std()

    # Apply SVD
    U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)

    # Get the principal components
    principal_components = np.dot(U, np.diag(S))

    # Create a DataFrame with the principal components
    pc_df = pd.DataFrame(data=principal_components[:, :2], columns=['PC1', 'PC2'])

    # Concatenate the principal components DataFrame with the original DataFrame
    # result_df = pd.concat([df, pc_df], axis=1)
  else:
    print("cols <2")
    pc_df= df.rename(columns={'col1': 'PC1', 'col2': 'PC2'})


  # Scatter plot of the data on the two principal components
  plt.figure(figsize=(10, 6))
  plt.scatter(pc_df['PC1'], pc_df['PC2'])
  plt.title('Scatter Plot of Data on Principal Components (SVD)')
  plt.xlabel('Principal Component 1 (PC1)')
  plt.ylabel('Principal Component 2 (PC2)')
  plt.grid(True)
  plt.show()

  return pc_df



class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights=None
        self.means=None
        self.covariances=None

        self.loglikelihood=[]
        self.bias_matrix = None



    def multivariate_normal_pdf(self, X, mean, covariance):
        n_features = X.shape[1]
        det = np.linalg.det(covariance)
        inv = np.linalg.inv(covariance)
        norm_const = 1.0 / ((2 * np.pi) ** (n_features / 2) * np.sqrt(det))
        exponent = -0.5 * np.sum(np.dot((X - mean), inv) * (X - mean), axis=1)
        return norm_const * np.exp(exponent)

    def new_expectation(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for i in range(self.n_components):
            mean = self.means[i]
            covariance = self.covariances[i]
            # print(i)
            weight = self.weights[i]
            # print(weight)


            # Compute the probability density function of each sample
            # pdf = multivariate_normal.pdf(X, mean, covariance)
            pdf = self.multivariate_normal_pdf(X,mean,covariance)

            # Compute the responsibility of each component for each sample
            responsibilities[:, i] = weight * pdf

        # Normalize the responsibilities
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        return responsibilities


    def log_likelihood(self,X):
        likelihood = 0

        for k in range(self.n_components):
            # likelihood += self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k]).reshape(-1)
            likelihood += self.weights[k] * self.multivariate_normal_pdf(X, self.means[k], self.covariances[k]).reshape(-1)
        return np.sum(np.log(likelihood))

    def fit(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.bias_matrix = np.diag(np.full(X.shape[1], 0.00000011))

        # np.random.seed(43)



        # Initialize parameters
        self.weights = np.ones(self.n_components) / self.n_components
        # self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.means = np.random.rand(self.n_components, X.shape[1])
        # self.covariances = np.array([np.cov(X.T) for _ in range(self.n_components)])


        self.covariances = np.array([np.identity(n_features) for _ in range(self.n_components)])


        # EM algorithm
        for i in range(self.max_iter):
            # E-step: Compute responsibilities
            prbs = self.new_expectation(X)

            # M-step: Update parameters
            # self.update_parameters(X, prbs)
            self.maximization(X,prbs)

            cur_logliklyhood=self.log_likelihood(X)

            # Check convergence
            # plt.scatter(_, sum)
            # self.loglikelihood.append(np.log(np.sum(new_responsibilities)))
            if i>0 and np.abs(cur_logliklyhood - loglikelihood) < self.tol:
                break
            loglikelihood=cur_logliklyhood


        # plt.show()
        return loglikelihood


    def fit_create_gif(self, X):


        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.bias_matrix = np.diag(np.full(X.shape[1], 0.00000011))

        # np.random.seed(43)



        # Initialize parameters
        self.weights = np.ones(self.n_components) / self.n_components
        # self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.means = np.random.rand(self.n_components, X.shape[1])
        # self.covariances = np.array([np.cov(X.T) for _ in range(self.n_components)])


        self.covariances = np.array([np.identity(n_features) for _ in range(self.n_components)])
        images = []

        for i in range(100):

            # E-step: Compute responsibilities
            prbs = self.new_expectation(X)

            # M-step: Update parameters
            # self.update_parameters(X, prbs)
            self.maximization(X,prbs)

            cur_logliklyhood=self.log_likelihood(X)

            # Check convergence
            # plt.scatter(_, sum)
            # self.loglikelihood.append(np.log(np.sum(new_responsibilities)))
            if i>0 and np.abs(cur_logliklyhood - loglikelihood) < self.tol:
                break
            loglikelihood=cur_logliklyhood
            # Plot contours and means

            # Create your plot here
            clustered_data = np.argmax(prbs, axis=1)
            # print(clustered_data)
            plt.scatter(X[:, 0], X[:, 1], c=clustered_data)

            # ellips
            # plt.scatter(X[:, 0], X[:, 1], s=0.5)

            # # Plot clusters and centers
            # for i in range((self.n_components)):
            #     mu = self.means[i]
            #     weight = self.weights[i]
            #     alpha = min(0.3, weight)
            #     plt.scatter(mu[0], mu[1], c='grey', zorder=10, s=100)

            #     x_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
            #     y_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
            #     X2, Y2 = np.meshgrid(x_range, y_range)

            #     plt.contourf(X2, Y2, prbs, alpha=alpha)



                # plt.contourf(*np.meshgrid(np.linspace(mu[0]-2, mu[0]+2, 100), np.linspace(mu[1]-2, mu[1]+2, 100)),
                #      prbs, alpha=alpha)
                # plt.contourf(*np.meshgrid(np.linspace(mu[0]-2, mu[0]+2, 100), np.linspace(mu[1]-2, mu[1]+2, 100)),
                            # clusters[i]['pdf'](np.meshgrid(np.linspace(mu[0]-2, mu[0]+2, 100), np.linspace(mu[1]-2, mu[1]+2, 100))), alpha=0.3)

            # Display the plot
            # display(plt.gcf())

            # Clear the current output for the next frame
            # clear_output(wait=True)

            # Save the plot as an image
            filename = f'plot_{i+1}.png'
            plt.savefig(filename)
            plt.close()

            # Append the image to the list
            images.append(imageio.imread(filename))

        # Save the list of images as a GIF file
        imageio.mimsave('/content/drive/MyDrive/Ml_sessonal/animation_trial.gif', images, duration=0.5)



    def compute_likelihood(self, X, mean, covariance):
        n_features = X.shape[1]
        det = np.linalg.det(covariance)
        inv = np.linalg.inv(covariance)
        # print(inv)
        exponent = -0.5 * np.sum((X - mean) @ inv * (X - mean), axis=1)
        likelihood = (1.0 / np.sqrt((2 * np.pi) ** n_features * det)) * np.exp(exponent)
        return (likelihood)


    def maximization(self,X, gamma):

      no_samples = X.shape[0]
      N_k = np.sum(gamma, axis=0)

      for k in range(self.n_components):
          self.means[k] = (1 / N_k[k]) * np.sum(gamma[:, k] * X.T, axis=1).T
          self.covariances[k] = (1 / N_k[k]) * np.dot(((X - self.means[k]).T * gamma[:, k]), (X - self.means[k]))
          self.covariances[k] += self.bias_matrix
          self.weights[k] = N_k[k] / no_samples



    def update_parameters(self, X, responsibilities):
        n_samples, _ = X.shape
        total_responsibilities = np.sum(responsibilities, axis=0)
        # print(responsibilities[:, 0].shape)
        self.weights = total_responsibilities / n_samples
        self.means = (responsibilities.T @ X) / total_responsibilities[:, np.newaxis]
        self.covariances = np.zeros((self.n_components, X.shape[1], X.shape[1]))

        for k in range(self.n_components):
            diff = X - self.means[k]
            # print(diff.shape)
            # self.covariances[k] = (diff.T @ (responsibilities[:, k] * diff)) / total_responsibilities[k]
            self.covariances[k] = (diff.T @ (responsibilities[:, k][:, np.newaxis] * diff)) / total_responsibilities[k]

def find_best_cluster(new_df):
  X=new_df.values

  logs=[]
  for i in range (2,10,1):
    log=-1000000000
    gmm = GaussianMixtureModel(n_components=i)
    for j in range (0,5,1):
      # gmm=GMM(n_components=5)
      cur_log=gmm.fit(X)
      if(cur_log>log):
        log=cur_log
    print(log)
    logs.append(log)

    # Print results clustered data

  start_value = 2
  x_values = list(range(start_value, start_value + len(logs)))
  print(logs)
  plt.plot(x_values,logs, label='My Line Plot')
  plt.savefig('/content/drive/MyDrive/Ml_sessonal/likelihood_plot.png')
  plt.show()


  diff=-100000
  ix=0
  for i in range(1,7):
    pre=logs[i-1]
    nxt=logs[i+1]
    d1 = abs(logs[i] - pre)
    d2 = abs(logs[i] - nxt)
    d=abs(d1-d2)
    if(diff<d):
      ix=i+start_value
      diff=d

  print("optimal number:")
  print(ix)
  return ix
  # responsibilities = gmm.new_expectation(X)
  # clustered_data = np.argmax(responsibilities, axis=1)
  # print(clustered_data)
  # plot the clustered data



  # plt.scatter(X[:, 0], X[:, 1], c=clustered_data)
  # plt.show()

def draw_opimal(n_components,X):
  gmm= GaussianMixtureModel(n_components=n_components)
  gmm.fit_create_gif(X)
  responsibilities = gmm.new_expectation(X)
  clustered_data = np.argmax(responsibilities, axis=1)
  print(clustered_data)
  plt.scatter(X[:, 0], X[:, 1], c=clustered_data)
  plt.savefig('/content/drive/MyDrive/Ml_sessonal/cluster_colors.png')
  plt.show()

def main():
  file_path6 = '/content/drive/MyDrive/Ml_sessonal/6D_data_points.txt'
  file_path3 = '/content/drive/MyDrive/Ml_sessonal/3D_data_points.txt'
  file_path21 = '/content/drive/MyDrive/Ml_sessonal/2D_data_points_1.txt'
  file_path22 = '/content/drive/MyDrive/Ml_sessonal/2D_data_points_2.txt'

  # df=read_data_2d(file_path21)
  # df=read_data_2d(file_path22)
  df=read_data_3d(file_path3)
  # df=read_data_6d(file_path6)

  new_df=apply_pca(df)
  x=find_best_cluster(new_df)
  draw_opimal(x,new_df.values)

main()

