
import numpy as np


def sigmoid(prediction):
    '''
    Sigmoid activation function, from logistic regression slides.
    '''
    return 1. / (1. + np.exp(-prediction))

def sigmoid_deriv(activation):
    derivative = activation*(1-activation)
    return derivative

def relu(prediction):
    out = np.copy(prediction)
    out[np.where(prediction < 0)]=0
    return out

def nooutact(prediction):
    return prediction

def relu_deriv(prediction):
    derivative = np.copy(prediction)
    derivative[np.where(prediction < 0)] = 0
    derivative[np.where(prediction >= 0)] = 1
    return derivative

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
    length = data.shape[1]
    prediction = prediction.ravel()
    data = data.ravel()
    calc = -data.dot(np.log(sigmoid(prediction)+ 1e-16)) - ((1 - data).dot(np.log(1 - sigmoid(prediction) + 1e-16)))
    norm = calc/length
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
