import numpy as np

import activation_functions


class MLP:

    def __init__(self, x, y, activation_function, eta, hiddenlayer1, *hiddenlayers):
        # Entry vector x
        self.x = x
        # Response vector (classes)
        self.Y = y
        # eta (n)
        self.eta = eta
        # declaring first layer at minimum
        self.number_of_layers=1
        # count number of hidden layers
        for sum in hiddenlayers:
            self.number_of_layers += 1
        # w array of amount layers (+1 for output layer)
        self.w = np.zeros(self.number_of_layers+1, dtype=object)
        # add first matrix of neurons into layer
        self.w[0] = np.zeros([hiddenlayer1, self.x.shape[0]], dtype=float)
        # assumption of reference, for create the next layer
        old = self.w[0]
        i=0
        for layer in hiddenlayers:
            i += 1
            #  instance of next hidden layer
            old = np.zeros([layer, old.shape[0]], dtype=float)
            # add in a layers array of matrices
            self.w[i] = old
        # add output layer to weight matrix
        self.w[-1] = np.zeros([ np.size(self.Y[0]), old.shape[0]], dtype=float)
        self.activation_function = activation_function

    def perceptron_node(self, x, wij):
        u = np.sum(x*wij)
        return self.activation_function(u)

    def perceptron_colum(self, x, wj):
        u = np.zeros(wj.shape[0])
        for i in range(np.size(u)):
            u[i] = self.perceptron_node(x, wj[i])
        return u

    def hidden_layer(self):
        u = np.zeros(self.number_of_layers,dtype=object)
        # first hidden layer result
        u[0] = self.perceptron_colum(self.x[:, 1], self.w[0])
        # next hidden layers result
        for i in range(1,np.size(u)):
            u[i] = self.perceptron_colum(u[i-1],self.w[i])
        # result layer
        y = self.perceptron_colum(u[-1],self.w[i])
        return activation_functions.rectifier_linear_derivative(y)
