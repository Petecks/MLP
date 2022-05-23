import numpy as np

import activation_functions as act

class MLP:

    def __init__(self, x, d, activation_function, eta, hiddenlayer1, *hiddenlayers):
        # Entry vector x
        self.x = x
        # Response vector (classes)
        self.d = d
        # eta (n)
        self.eta = eta
        # declaring first layer at minimum
        self.number_of_layers=1
        # count number of hidden layers
        for sum in hiddenlayers:
            self.number_of_layers += 1
        # w array of amount layers (+1 for output layer)
        self.w = np.ones(self.number_of_layers+1, dtype=object)
        # add first matrix of neurons into layer
        self.w[0] = np.ones([hiddenlayer1, self.x.shape[0]], dtype=float)
        # assumption of reference, for create the next layer
        old = self.w[0]
        i=0
        for layer in hiddenlayers:
            i += 1
            #  instance of next hidden layer
            old = np.ones([layer, old.shape[0]], dtype=float)
            # add in a layers array of matrices
            self.w[i] = old
        # add output layer to weight matrix
        self.w[-1] = np.ones([ np.size(self.d[0]), old.shape[0]], dtype=float)
        self.error = self.w.copy()
        self.v = self.w.copy()
        self.activation_function = activation_function

    # each node sum and activation function
    def perceptron_node(self, x, wij):
        u = np.sum(x*wij)
        return self.activation_function(u), u

    # each colum layer of layers
    def perceptron_colum(self, x, wj):
        u = np.zeros(wj.shape[0])
        v = np.zeros(wj.shape[0])
        for i in range(np.size(u)):
            u[i], self.v[i] = self.perceptron_node(x, wj[i])
        return u,v

    # softmax probability implementation
    def softmax(self,y):
        return np.exp(y)/np.sum(np.exp(y))

    def test_model_epoch(self):
        u = np.zeros(self.number_of_layers,dtype=object)
        # first hidden layer result
        u[0], self.v[0] = self.perceptron_colum(self.x[0,:], self.w[0])
        # next hidden layers result
        for i in range(1,np.size(u)):
            u[i],self.v = self.perceptron_colum(u[i-1],self.w[i])
        # result layer
        y = self.perceptron_colum(u[-1],self.w[-1])
        # probability vector output
        return y
        # add in other option
        # return self.softmax(y)

    def training_mode(self):
        output = self.test_model_epoch()
        self.back_propagation(output)



    # error = (output - expected) * transfer_derivative(output)
    # error = (weight_k * error_j) * transfer_derivative(output)
    # weight = weight - learning_rate * error * input

    # Back propagation of error to calculate the new weights
    def back_propagation(self,output):
        # output error calculation
        self.error[-1] = (output - self.d[0]) * act.derivative_sigmoid(output)

        for i in reversed(range(np.size(self.error)-1)):
            self.error[i] = (self.w[i+1] + self.error[i+1]) * act.derivative_sigmoid(output)
        # print(self.w[-1])
        # self.w[-1] = self.w[-1] - (self.eta * output_error * self.x[0])
        # print(self.w[-1])
        # hidden layers error calculation
        # hidden_error = (weight_k * error_j) * act.derivative_sigmoid(output)

