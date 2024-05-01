# -*- coding: utf-8 -*-
"""offline3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19XffJspIe-Y9OegY03z0kMeQeTOjcgbt
"""

import pandas as pd
import numpy as np
import torchvision.datasets as ds
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, confusion_matrix

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

def get_train_data():
  train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

  num_samples = len(train_validation_dataset)
  num_train_samples = int(0.85 * num_samples)
  num_valid_samples = num_samples - num_train_samples

  # Split the dataset into training and validation sets
  train_dataset, valid_dataset = torch.utils.data.random_split(train_validation_dataset,
                                                              [num_train_samples, num_valid_samples])

  # Get the class labels
  tr_labels = train_dataset.dataset.classes
  val_labels=valid_dataset.dataset.classes
  return train_dataset,valid_dataset,tr_labels,val_labels

def flatten_train_image(validation_dataset,labels):
  train_images=[]
  train_labels=[]
  for i in range(0,len(validation_dataset)):
    image,l=validation_dataset[i]
    flattened_array = image[0].view(-1).numpy().reshape(1, -1)
    train_images.append(flattened_array)
    train_labels.append(labels[l])


  return train_images,train_labels

def train_data_prep():
  tr_data,val_data,tr_labels,v_labels=get_train_data()

  train_images,train_labels=flatten_train_image(tr_data,tr_labels)
  val_images,val_labels=flatten_train_image(val_data,v_labels)

  tr_im_array = np.vstack(train_images)
  val_im_array = np.vstack(val_images)

  x_train=tr_im_array.T
  x_val=val_im_array.T

  label_binarizer = LabelBinarizer()
  y_train =  label_binarizer.fit_transform(train_labels).T
  y_val = label_binarizer.fit_transform(val_labels).T

  return x_train,y_train,x_val,y_val

"""model"""

x_train,y_train,x_val,y_val=train_data_prep()
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

"""test......................

Lter
"""

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input,training):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Softmax:
    def forward(self, input):
        # print('sf')
        tmp = np.exp(input)
        output = tmp / np.sum(tmp, axis=0)
        return output

    def backward(self, x):
        return x

    # def forward(self, input):
    #     tmp = np.exp(input)
    #     self.output = tmp / np.sum(tmp)
    #     return self.output

    # def backward(self, output_gradient, learning_rate):
    #     # This version is faster than the one presented in the video
    #     n = np.size(self.output)
    #     return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)

class ReLU:
    def forward(self,x):
        # print('rf')
        return np.maximum(0, x)

    def backward(self,x):
        # print('rb')
        return np.where(x > 0, 1, 0)


    # def forward(self, input):
    #     return np.maximum(0, input)

    # def backward(self, output_gradient, learning_rate):
    #     return np.where(input > 0, 1, 0)

"""my own

"""

def cost_function(y_pred, y_true):
    m = y_true.shape[1]
    epsilon=1e-15

    cost = -(1/m)*np.sum(y_true*np.log(y_pred))
    return cost
def cost_prime(y_pred, y_true):
    return y_pred-y_true

class Dense(Layer):

    def __init__(self, input_size, output_size,ntype):
        np.random.seed(63)
        self.weights = np.random.randn(output_size, input_size)*0.1
        self.bias = np.zeros((output_size, 1))
        self.m=None
        self.ntype=ntype
        # for adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.s_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.s_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
        self.t = 0


    def forward(self, input,training):
        self.m=input.shape[1]
        # print("xshape=",self.m)
        self.input = input
        z= np.dot(self.weights, self.input) + self.bias
        # print(z)
        ac=None
        if(self.ntype == 'h'):
          # relu
          ac=ReLU()
        else:
          # softmax
          ac=Softmax()

        a=ac.forward(z)
        return a

    def backward(self, output_gradient, learning_rate):
        weights_gradient=None
        bias_gradient=None
        input_gradient=None

        act=ReLU()

        # if(type == 'h'):
        #   weights_gradient = np.dot(output_gradient, self.input.T)
        #   bias_gradient = (1/m)*np.sum(output_gradient, axis = 1, keepdims = True)
        #   input_gradient = np.dot(self.weights.T, output_gradient)

        #   return input_gradient

        # else:
        #   # dz2  out_gr
        weights_gradient = (1/self.m)*np.dot(output_gradient, self.input.T)
        bias_gradient = (1/self.m)*np.sum(output_gradient, axis = 1, keepdims = True)
        input_gradient = (1/self.m)*np.dot(self.weights.T, output_gradient)*act.backward(self.input)

        # for adam
        self.v_w = self.beta1*self.v_w+(1-self.beta1)*weights_gradient
        self.v_b = self.beta1*self.v_b+(1-self.beta1)*bias_gradient

        self.s_w = self.beta2*self.s_w+(1-self.s_w)*(weights_gradient**2)
        self.s_b = self.beta2*self.s_b+(1-self.s_b)*(bias_gradient**2)

        weight_update = self.v_w/(np.sqrt(self.s_w + self.epsilon))
        bias_update = self.v_b/(np.sqrt(self.s_b + self.epsilon))

        self.weights -= learning_rate * weight_update
        self.bias -= learning_rate * bias_update

        return input_gradient

"""under later"""

def test_predict(network,input):
    output = input
    for layer in network:
        output = layer.forward(output,training=False)
    return output

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output,training=True)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    errors=[]
    for e in range(epochs):
        error = 0

        # forward
        output = predict(network, x_train)

        # error
        error += loss(output,y_train)

        # backward
        grad = loss_prime(output,y_train)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

        # error /= len(x_train)
        # if verbose:
        #     print(f"{e + 1}/{epochs}, error={error}")
        errors.append(error)
    return errors,network

class Dropout(Layer):

    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def forward(self, input, training):
        # print('from drop out',training)
        if training:
            self.dropout_mask = (np.random.rand(*input.shape) < (1 - self.dropout_rate))
            input *= self.dropout_mask / (1 - self.dropout_rate)
        return input

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.dropout_mask / (1 - self.dropout_rate)

class DropoutLayer(Layer):

    def __init__(self, input_size, output_size, ntype, dropout_rate):
        # np.random.seed(63)
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.bias = np.zeros((output_size, 1))
        self.m = None
        self.ntype = ntype
        self.dropout_rate=dropout_rate
        self.dropout_layer = Dropout(dropout_rate)
        # for adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.s_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.s_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
        self.t = 0

    def forward(self, input, training):
        self.m = input.shape[1]
        self.input = input
        z = np.dot(self.weights, input) + self.bias

        ac = ReLU() if (self.ntype == 'h') else Softmax()
        a = ac.forward(z)

        # Assuming Dropout class is correctly implemented
        a = self.dropout_layer.forward(a, training=training)
        return a

    def backward(self, output_gradient, learning_rate):
        weights_gradient = None
        bias_gradient = None
        input_gradient = None

        act = ReLU() if (self.ntype == 'h') else Softmax()

        weights_gradient = (1/self.m) * np.dot(output_gradient, self.input.T)
        bias_gradient = (1/self.m) * np.sum(output_gradient, axis=1, keepdims=True)
        input_gradient = (1/self.m) * np.dot(self.weights.T, output_gradient) * act.backward(self.input)

        input_gradient = self.dropout_layer.backward(input_gradient, learning_rate)

        self.t += 1
        self.v_w = self.beta1 * self.v_w + (1 - self.beta1) * weights_gradient
        self.v_b = self.beta1 * self.v_b + (1 - self.beta1) * bias_gradient

        self.s_w = self.beta2 * self.s_w + (1 - self.beta2) * (weights_gradient**2)
        self.s_b = self.beta2 * self.s_b + (1 - self.beta2) * (bias_gradient**2)

        # Corrected Adam optimizer updates
        weight_update = self.v_w / (np.sqrt(self.s_w) + self.epsilon)
        bias_update = self.v_b / (np.sqrt(self.s_b) + self.epsilon)

        self.weights -= learning_rate * weight_update
        self.bias -= learning_rate * bias_update

        return input_gradient

def model(x_train,y_train,batch_size,epochs):
  # X = x_train[:,:10000]
  # Y = y_train[:,:10000]

  tr_network = [
      Dense(784, 512,'h'),
      DropoutLayer(512,512,'h',0.3),
      Dense(512, 26,'o')
  ]

  # train
  if ( x_train.shape[1] % batch_size != 0):
    return 0
  final_errors=[]
  tr_loss=[]
  val_loss=[]
  tr_acc=[]
  val_acc=[]
  val_f1=[]
  for e in range(epochs):
    batches = x_train.shape[1] / batch_size
    batches_int = int(batches)
    batch_errors=[]
    for b in range(batches_int):
      X = x_train[:,b*batch_size:(b+1)*batch_size]
      Y = y_train[:,b*batch_size:(b+1)*batch_size]

      errors,tr_network=train(tr_network, cost_function, cost_prime, X, Y, epochs=1, learning_rate=0.005)
      # batch_error=np.mean(errors)


    # print(f"epoch_error error {e}/100",epoch_error)
    # final_errors.append(epoch_error)

    t_ac,t_f1,t_l=loop_accuracy(x_train,y_train,tr_network)
    v_ac,v_f1,v_l=loop_accuracy(x_train,y_train,tr_network)

    tr_loss.append(t_l)
    tr_acc.append(t_ac)
    val_acc.append(v_ac)
    val_loss.append(v_l)
    val_f1.append(v_f1)

  # decision boundary plot

  plt.plot(tr_acc, label='Training Acc')
  plt.xlabel('Data Point')
  plt.ylabel('Tr Acc')
  plt.title('Training Acc')
  plt.show()

  plt.plot(tr_loss, label='tr loss')
  plt.xlabel('Data Point')
  plt.ylabel('Tr Loss')
  plt.title('Line Graph of Loss')
  plt.show()

  plt.plot(val_acc, label='Val Acc')
  plt.xlabel('Data Point')
  plt.ylabel('Val Acc')
  plt.title('Line Graph of val Acc')
  plt.show()

  plt.plot(val_loss, label='val loss')
  plt.xlabel('Data Point')
  plt.ylabel('val loss')
  plt.title('val loss')
  plt.show()

  plt.plot(val_f1, label='val F1')
  plt.xlabel('Data Point')
  plt.ylabel('val F1')
  plt.title('Line Graph of val f1')
  plt.show()

  return tr_network

def loop_accuracy(inp, labels, network):
    output = test_predict(network, inp)

    l = cost_function(output,y_train)
    a_out = np.argmax(output, axis=0)
    labels = np.argmax(labels, axis=0)

    acc = np.mean(a_out == labels) * 100

    # Compute F1 score
    f1 = f1_score(labels, a_out, average='weighted')

    return acc,f1,l

tr_net=model(x_train,y_train,6240,50)

def accuracy(inp, labels, network):
    output = test_predict(network, inp)

    a_out = np.argmax(output, axis=0)
    labels = np.argmax(labels, axis=0)

    acc = np.mean(a_out == labels) * 100

    # Compute F1 score
    f1 = f1_score(labels, a_out, average='weighted')

    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels, a_out)

    return acc, f1, conf_matrix

X=x_val
Y=y_val
acc,f1,conf_matrix=accuracy(X,Y,tr_net)
print(acc)
print(f1)
print(conf_matrix)

import pickle
def export_model(model):
  weights_biases = []
  for layer in model:
      if isinstance(layer, Dense):
          layer_params = {
              'weights': layer.weights,
              'bias': layer.bias,
              'ntype':layer.ntype

          }
          weights_biases.append(layer_params)
      elif isinstance(layer, DropoutLayer):
          layer_params = {
              'weights': layer.weights,
              'bias': layer.bias,
              'dropout_rate': layer.dropout_rate,
              'ntype':layer.ntype
          }
          weights_biases.append(layer_params)

  with open('/content/drive/MyDrive/Ml_sessonal/1805063.pkl', 'wb') as file:
      pickle.dump(weights_biases, file)

export_model(tr_net)

