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
        # w array of amount layers
        self.w = np.zeros(self.number_of_layers, dtype=object)
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
        u = np.zeros(self.w.shape)

        #first hidden layer
        u[:,0] = self.perceptron_colum(self.x,self.w[:,0])
        # each hidden layer
        for i in range(0,self.w.shape[1]-1):
            u[:,i+1] = self.perceptron_colum(u[:i], self.w[:,i+1])
        # result layer
        y = np.zeros(self.Y.shape[1])
        y = self.perceptron_colum(u[:-1], np.ones(np.size(self.Y)))
        return activation_functions.rectifier_linear_derivative(y)