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
        # self.w[0] = np.ones([hiddenlayer1, self.x.shape[1]], dtype=float)
        # assumption of reference, for create the next layer
        old = self.w[0]
        i=0
        for layer in hiddenlayers:
            i += 1
            #  instance of next hidden layer
            old = -.5 + np.random.rand(layer, old.shape[0])
            # old = np.ones([layer, old.shape[0]], dtype=float)
            # add in a layers array of matrices
            self.w[i] = old
        # add output layer to weight matrix
        # self.w[-1] = np.ones([ np.size(self.d[0]), old.shape[0]], dtype=float)
        # self.error = self.w.copy()*0
        # self.v = self.w.copy()*0
        # self.y = self.w.copy()*0
        self.error = np.zeros(self.number_of_layers, dtype=object)
        self.v = np.zeros(self.number_of_layers, dtype=object)
        self.y = np.zeros(self.number_of_layers, dtype=object)
        self.activation_function = activation_function

    # each node sum and activation function
    def perceptron_node(self, x, wij):
        v = np.sum(x*wij)
        return self.activation_function(v),v

    # each colum layer of layers
    def perceptron_colum(self, x, wj):
        u = np.zeros(wj.shape[0])
        v = np.zeros(wj.shape[0])
        for i in range(np.size(u)):
            u[i],v[i] = self.perceptron_node(x, wj[i])
        return u,v

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
        # y = self.perceptron_colum(u[-1],self.w[-1])
        # probability vector output
        if flag == 1:
            return self.softmax(y[-1])
        return y[-1]

        # add in other option




    def training_mode(self):
        for i in range(self.n_max):
            for j in range(self.x.shape[0]):
                output = self.test_model_epoch(j,0)
                self.back_propagation(j,output)



    # error = (output - expected) * transfer_derivative(output)
    # error = (weight_k * error_j) * transfer_derivative(output)
    # weight = weight - learning_rate * error * input

    # Back propagation of error to calculate the new weights
    def back_propagation(self,j,output):
        # output error calculation
        shaped_d = np.zeros(np.shape(output))
        shaped_d[self.d[j]] = 1

        self.error[-1] = ( shaped_d - self.y[-1]) * act.derivative_sigmoid(self.v[-1])
        self.error[-1] = np.ones((len(self.error[-1]),len(self.y[-2]))) * self.error[-1].reshape((len(self.error[-1]), 1))
        delta_w = self.eta * self.error[-1] *  np.ones((len(self.error[-1]), len(self.y[-2])))*self.y[-2]

        self.w[-1] = self.w[-1] + delta_w

        for i in reversed(range(np.size(self.error)-1)):
            self.error[i] = np.sum(self.w[i + 1] * self.error[i + 1]) * act.derivative_sigmoid(self.v[i])
            if i==0:
                self.error[i] = np.ones((len(self.error[i]), len(self.x[0]))) * self.error[i].reshape((len(self.error[i]), 1))
                delta_w = self.eta * self.error[i] * np.ones((len(self.error[i]), len(self.x[0]))) * self.x[0]
                self.w[i] = self.w[i] + delta_w
                return
            self.error[i] = np.ones((len(self.error[i]), len(self.y[i-1]))) *self.error[i].reshape((len(self.error[i]), 1))
            delta_w = self.eta * self.error[i] * np.ones((len(self.error[i]), len(self.y[i-1]))) * self.y[i-1]
            self.w[i] = self.w[i] + delta_w

        # print(self.w[-1])
        # self.w[-1] = self.w[-1] - (self.eta * output_error * self.x[0])
        # print(self.w[-1])
        # hidden layers error calculation
        # hidden_error = (weight_k * error_j) * act.derivative_sigmoid(output)

