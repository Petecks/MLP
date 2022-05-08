import numpy as np

def sigmoid(x):
     return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
     return x/(1-x)

def rectifier_linear(x):
     return x if x > 0 else 0

def rectifier_linear_derivative(x):
     return 1 if x > 0 else 0