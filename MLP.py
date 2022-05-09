import numpy as np

import activation_functions


class MLP:

    def __init__(self, x, y, activation_function, hidden_r=2, hidden_c=2, eta=0.2):

        self.x = x
        self.Y = y
        self.eta = eta
        self.w = np.zeros([hidden_r, hidden_c], dtype=float)
        self.activation_function = activation_function

    def perceptron_node(self, x, wij):
        u = 0
        for i in range(np.size(x)):
                u += np.sum(x*wij)
        return self.activation_function(u)

    def perceptron_column(self, x, wj):
        u = np.zeros(np.size(wj))
        for i in range(np.size(wj)):
            u[i] = self.perceptron_node(x, wj[i])
        return u

    def hidden_layer(self):
        u = np.zeros(self.w.shape)

        #first hidden layer
        u[:,0] = self.perceptron_column(self.x,self.w[:,0])
        # each hidden layer
        for i in range(0,self.w.shape[1]-1):
            u[:,i+1] = self.perceptron_column(u[:i], self.w[:,i+1])
        # result layer
        y = np.zeros(self.Y.shape[1])
        y = self.perceptron_column(u[:-1], np.ones(np.size(self.Y)))
        return activation_functions.rectifier_linear_derivative(y)
