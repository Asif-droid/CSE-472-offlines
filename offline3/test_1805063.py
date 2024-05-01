import pandas as pd
import numpy as np
import torchvision.datasets as ds
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, confusion_matrix
import pickle

def flatten_train_image(validation_dataset,labels):
  train_images=[]
  train_labels=[]
  for i in range(0,len(validation_dataset)):
    image,l=validation_dataset[i]
    flattened_array = image[0].view(-1).numpy().reshape(1, -1)
    train_images.append(flattened_array)
    train_labels.append(labels[l])


  return train_images,train_labels

def test_predict(network,input):
    output = input
    for layer in network:
        output = layer.forward(output,training=False)
    return output

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
# 

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


class Softmax:
    def forward(self, input):
        # print('sf')
        tmp = np.exp(input)
        output = tmp / np.sum(tmp, axis=0)
        return output

    def backward(self, x):
        return x


class ReLU:
    def forward(self,x):
        # print('rf')
        return np.maximum(0, x)

    def backward(self,x):
        # print('rb')
        return np.where(x > 0, 1, 0)


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





def load_model(path):
  weights_biases=None
  with open(path, 'rb') as file:
      weights_biases = pickle.load(file)


  loaded_network = []
  for params in weights_biases:
      if 'dropout_rate' in params:
          # Recreate Dense layer
          layer = DropoutLayer(params['weights'].shape[0], params['weights'].shape[1],params['ntype'],params['dropout_rate'])
          layer.weights = params['weights']
          layer.bias = params['bias']
          loaded_network.append(layer)
      else:
          # Recreate DropoutLayer
          layer = Dense(params['weights'].shape[0], params['weights'].shape[1],params['ntype'])
          layer.weights = params['weights']
          layer.bias = params['bias']
          loaded_network.append(layer)

  return loaded_network

def get_test_data():
  independent_test_dataset =ds.EMNIST(root='./data',
                       split='letters',
                             train=False,
                             transform=transforms.ToTensor())
  # train_validation_dataset
  labels=independent_test_dataset.classes
  return independent_test_dataset,labels

def test_data_prep():
  test_validation_dataset,labels=get_test_data()
  test_images,train_labels=flatten_train_image(test_validation_dataset,labels)
  tr_im_array = np.vstack(test_images)
  x_train=tr_im_array.T
  label_binarizer = LabelBinarizer()
  y_train =  label_binarizer.fit_transform(train_labels).T

  return x_train,y_train

def test():
  path='/content/drive/MyDrive/Ml_sessonal/1805063.pkl'
  model = load_model(path)
  x_test,y_test=test_data_prep()
  X=x_test
  Y=y_test
  acc,f1,conf_matrix=accuracy(X,Y,model)
  print(acc)
  print(f1)
  print(conf_matrix)

test()