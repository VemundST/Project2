import numpy as np
from scipy.stats import t
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import scipy.linalg as scl



''' Franke Function and Design Matrix '''
def FrankeFunction(x, y, noise_level=0):
    '''
    Usage: Computes the Franke function with user defined spatial paramters and noise
    Input:  x,y = meshgrid of spatial vectors
            noise_level = scalar value
    Output: Franke function = matrix of size [x]
    '''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = noise_level*np.random.randn(len(x),len(y))
    return term1 + term2 + term3 + term4 + noise


def DesignDesign(x, y, power,ravel=False):
    '''
    This function employs the underlying pattern governing a design matrix
    on the form [1,x,y,x**2,x*y,y**2,x**3,(x**2)*y,x*(y**2),y**3 ....]

    x_power=[0,1,0,2,1,0,3,2,1,0,4,3,2,1,0,...,n,n-1,...,1,0]
    y_power=[0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,...,0,1,...,n-1,n]

    input:  x,y = 1D spatial vectors, or raveled mesh if ravel=True
            power = polynomial degree
            ravel = False (input vectors will not be meshed and raveled)
                  = True (input vecotors wil be kept as is)
    output: DesignMatrix = The design matrix
    '''

    concat_x   = np.array([0,0])
    concat_y   = np.array([0,0])


    for i in range(power):
        toconcat_x = np.arange(i+1,-1,-1)
        toconcat_y = np.arange(0,i+2,1)
        concat_x   = np.concatenate((concat_x,toconcat_x))
        concat_y   = np.concatenate((concat_y,toconcat_y))

    concat_x     = concat_x[1:len(concat_x)]
    concat_y     = concat_y[1:len(concat_y)]

    if ravel:
        X = x
        Y = y
    else:
        X,Y          = np.meshgrid(x,y)
        X            = np.ravel(X)
        Y            = np.ravel(Y)
    DesignMatrix = np.empty((len(X),len(concat_x)))
    for i in range(len(concat_x)):
        DesignMatrix[:,i]   = (X**concat_x[i])*(Y**concat_y[i])
    return DesignMatrix


''' Error Metrics'''

def MSE(y, ytilde):
    '''
    Usage: calculates the Mean Squared Error
    Input:  y = observed data
            ytilde = predicted data
    output: Mean Squared Error
    '''
    return (np.sum((y-ytilde)**2))/y.size


def R2Score(y, ytilde):
    '''
    Usage: calculates the R2-score
    Input:  y = observed data
            ytilde = predicted data
    output: R2score
    '''
    return 1 - np.sum((y - ytilde) ** 2) / np.sum((y - np.mean(ytilde)) ** 2)


def confidence_interval(design, sigma, confidence, _lambda=0):
    '''
    Usage: calculates the confidence interval for OLS and Lasso if noise is known
    Input:  design = user defined design matrix
            sigma = a-priori known noise level
            confidence = 0.95 = 95 perecent confidence level
            _lambda = if lambda = 0, calculates confidence for OLS
                      if lambda ~=0, calculates confidence for Ridge
    output: Confidence interval for all beta values.
    '''
    if _lambda != 0:
        I=np.eye(design.shape[1])
        inverse_term   = np.linalg.inv(design.T.dot(design) + _lambda*I)
        variance_mat   = sigma**2*(inverse_term)*(design.T.dot(design))*(inverse_term)
    else:
        inverse_term   = np.linalg.inv(design.T.dot(design))
        variance_mat   = inverse_term*sigma**2

    standard_dev   = np.sqrt(np.diag(variance_mat))
    return standard_dev*norm.ppf(confidence+(1-confidence)/2)
