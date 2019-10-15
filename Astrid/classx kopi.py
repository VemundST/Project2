## Functions
import numpy as np
from scipy.stats import t
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import scipy.linalg as scl
import utilities as utils
from sklearn.linear_model import Lasso

''' Regression Algorithms'''

class OLS:
    def __init__(self, _lambda=0):
        self._lambda = 0

    def fit(self, design, data):
        inverse_term = np.linalg.inv(design.T.dot(design))
        self.coef_= inverse_term.dot(design.T).dot(data)
        return self.coef_

    def predict(self, design):
        self.pred_ = design@self.coef_
        return self.pred_


class Ridge:
    def __init__(self, _lambda=0):
        self._lambda = _lambda

    def fit(self, design, data):
        inverse_term = np.linalg.inv(design.T.dot(design) + self._lambda*np.eye((design.shape[1])))
        self.coef_ = inverse_term.dot(design.T).dot(data)
        return self.coef_

    def predict(self, design):
        self.pred_ = design@self.coef_
        return self.pred_

class Lasso_:
    def __init__(self, _lambda=0, intcept=True, max_it=10e4):
        self._lambda = _lambda
        self.intcept = intcept
        self.max_it = max_it
    def fit(self, design, data):
        self.coef_ = Lasso(alpha=self._lambda, fit_intercept = self.intcept, max_iter = self.max_it).fit(design, data).coef_
        return self.coef_

    def predict(self, design):
        self.pred_ = design@self.coef_
        return self.pred_
