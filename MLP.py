import numpy as np

import activation_functions as act

class MLP:
    # plus output layer
    def __init__(self, x, d, activation_function, eta, n_max, hiddenlayer1, *hiddenlayers):
        # Entry vector x
        self.x = x
        self.x = np.append(self.x, np.ones([self.x.shape[0],1]),axis=1)
        # Response vector (classes)
        self.d = d
        # eta (n)
        self.eta = eta
        # declaring first layer at minimum
        self.n_max = n_max

        self.number_of_layers=1
        # count number of hidden layers
        for sum in hiddenlayers:
            self.number_of_layers += 1
        # w array of amount layers
        self.w = np.ones(self.number_of_layers, dtype=object)
        # add first matrix of neurons into layer
        self.w[0] = -.5 + np.random.rand(hiddenlayer1, self.x.shape[1])
        # assumption of reference, for create the next layer
        old = self.w[0]
        i=0
        for layer in hiddenlayers:
            i += 1
            #  instance of next hidden layer
            old = -.5 + np.random.rand(layer, old.shape[0])
            self.w[i] = old
        # add output layer to weight matrix

        self.error = np.zeros(self.number_of_layers, dtype=object)
        self.v = np.zeros(self.number_of_layers, dtype=object)
        self.y = np.zeros(self.number_of_layers, dtype=object)
        self.activation_function = activation_function
        self.error_count = np.zeros(n_max)
    # each node sum and activation function
    def perceptron_node(self, x, wij):
        v = np.sum(x*wij)
        return self.activation_function(v),v

    # each colum layer of layers
    def perceptron_colum(self, x, wj):
        y = np.zeros(wj.shape[0])
        v = np.zeros(wj.shape[0])
        for i in range(np.size(v)):
            y[i],v[i] = self.perceptron_node(x, wj[i])
        return y,v

    # softmax probability implementation
    def softmax(self,y):
        return np.exp(y)/np.sum(np.exp(y))

    def test_model_epoch(self,n,flag=0):
        y = np.zeros(self.number_of_layers,dtype=object)
        v = np.zeros(self.number_of_layers,dtype=object)
        # first hidden layer result
        y[0],v[0]= self.perceptron_colum(self.x[n,:], self.w[0])
        # next hidden layers result
        for i in range(1,np.size(y)):
            y[i],v[i] = self.perceptron_colum(y[i-1],self.w[i])
        # result layer
        self.v = v
        self.y = y
        # probability vector output
        if flag == 1:
            return self.softmax(y[-1])
        return y[-1]

    def training_mode(self):
        for i in range(self.n_max):
            for j in range(self.x.shape[0]): #sem holdout
            # for j in range(105): # holdout model
                self.test_model_epoch(j,0)
                self.back_propagation(j,i)

    # Back propagation of error to calculate the new weights
    def back_propagation(self,j,epoch):
        # output error calculation
        shaped_d = np.zeros(np.shape(self.y[-1]))
        shaped_d[self.d[j]] = 1
        max_index_d = np.argmax(shaped_d)
        max_index_y = np.argmax(self.y[-1])
        # Compare errors to sum
        if max_index_d != max_index_y:
            self.error_count[epoch] += 1

        #outlayer calculation weights
        self.error[-1] = ( shaped_d - self.y[-1]) * act.derivative_sigmoid(self.v[-1])
        self.error[-1] = np.ones((len(self.error[-1]),len(self.y[-2]))) * self.error[-1].reshape((len(self.error[-1]), 1))
        delta_w = self.eta * self.error[-1] * np.ones((len(self.error[-1]), len(self.y[-2]))) * self.y[-2]

        self.w[-1] = self.w[-1] + delta_w
        # hidden layers calculation weights
        for i in reversed(range(np.size(self.error)-1)):
            self.error[i] = [sum(x) for x in zip(*(self.w[i + 1] * self.error[i + 1]))] * act.derivative_sigmoid(self.v[i])
            # if layer is the input layer
            if i==0:
                self.error[i] = np.ones((len(self.error[i]), len(self.x[j]))) * self.error[i].reshape((len(self.error[i]), 1))
                delta_w = self.eta * self.error[i] * np.ones((len(self.error[i]), len(self.x[j]))) * self.x[j]
                self.w[i] = self.w[i] + delta_w
                return
            self.error[i] = np.ones((len(self.error[i]), len(self.y[i-1]))) *self.error[i].reshape((len(self.error[i]), 1))
            delta_w = self.eta * self.error[i] * np.ones((len(self.error[i]), len(self.y[i-1]))) * self.y[i-1]
            self.w[i] = self.w[i] + delta_w
