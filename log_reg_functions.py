
import numpy as np


def sigmoid(prediction):
    '''
    Sigmoid activation function, from logistic regression slides.
    '''
    return 1. / (1. + np.exp(-prediction))

def relu_(prediction):
    '''
    Relu activation function
    '''
    return prediction

def cost_mse_ols(design, data, beta):
    '''
    Mean squared error
    '''
    return (data - design.dot(beta)).T*(data - design.dot(beta))

def cost_grad_ols(design, data, beta):
    '''
    Calculates the first derivative of MSE w.r.t beta.
    '''
    return (2/len(data))*design.T.dot(design.dot(beta)-data) #logistic regression slides

def cost_log_ols(prediction, data):
    '''
    Logisitic regression cost function
    '''
    calc = -data.dot(np.log(sigmoid(prediction)+ 1e-16)) - ((1 - data).dot(np.log(1 - sigmoid(prediction) + 1e-16)))
    norm = calc/data.shape[1]
    return norm

def gradient_ols(design, data, p):
    '''
    Gradient w.r.t log
    '''
    return np.dot(design.T, (p - data)) / data.shape[0]


def gradient_solver(N, eta, design, data, beta=None):
    M=len(data)
    if beta != None:
        beta = beta
    else:
        beta = np.random.randn(design.shape[1])

    for i in range(N):
        gradients = cost_grad_ols(design,frank,beta)
        beta -= eta*gradients
    return beta
