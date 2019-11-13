import numpy as np
'''
This file includes the activation functions and cost functions used by
NeuralNetwork.py
and
LogisticRegression.py
'''

def sigmoid(prediction):
    '''
    Sigmoid activation function, from logistic regression slides.
    '''
    return 1. / (1. + np.exp(-prediction))

def sigmoid_deriv(activation):
    '''
    Returns derivative of sigmoid activation function.
    '''
    derivative = activation*(1-activation)
    return derivative

def relu(prediction):
    '''
    ReLU activation function.
    '''
    out = np.copy(prediction)
    out[np.where(prediction < 0)]=0
    out = np.clip(out,-300,300)
    return out

def relu_deriv(prediction):
    '''
    Returns the derivative of ReLU.
    '''
    derivative = np.copy(prediction)
    derivative[np.where(prediction < 0)] = 0
    derivative[np.where(prediction >= 0)] = 1
    return derivative

def nooutact(prediction):
    '''
    Can be used for activation in output layer in case of regression.
    '''
    return prediction

def nooutact_deriv(prediction):
    out = np.ones(prediction.shape)
    return out



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


def reshaper(k, data):
    '''
    Usage: Manages the data for k_fold_cv
    Input: k = number of folds
           data = shuffled input data or input design matrix
    output: Splitted data
    '''
    output = []
    j = int(np.ceil(len(data)/k))
    for i in range(k):
        if i<k:
            output.append(data[i*j:(i+1)*j])
        else:
            output.append(data[i*j:])
    return np.asarray(output)


def k_fold_reshaper(k, indata, indesign, shuffle=True):

    '''
    Usage: k-fold cross validation employing either RidgeRegression, OridinaryLeastSquares or ols_svd
    Input: k = number of folds
           indata = datapoints
           indesign = user defined design matrix
           predictor = RidgeRegression, OridinaryLeastSquares or ols_svd
           _lambda = hyperparameter/penalty paramter/tuning parameter for RidgeRegression
           shuffle = False, input data will not be shuffled
                     True, input data will be shuffled
    output: r2_out/k = averaged out sample R2-score
            mse_out/k = averaged out sample MSE
            r2_in/k = averaged in sample R2-Score
            mse_in/k = averaged in sample MSE
    '''
    mask = np.arange(indata.shape[0])
    if shuffle:
        np.random.shuffle(mask)
    data = reshaper(k, indata[mask])
    design = reshaper(k, indesign[mask])
    return data,design
