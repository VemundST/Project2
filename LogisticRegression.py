#logistic regression
import numpy as np
import matplotlib.pyplot as plt
import log_reg_functions as lrf



class LOGREG():
    def __init__(self, eta=0.1, early_stop_tol = 0.0, early_stop_nochange=10, doplot = False, doprint=False):
        self.eta=eta
        self.early_stop_tol = early_stop_tol
        self.early_stop_nochange = early_stop_nochange
        self.beta = float()
        self.costvec=[]
        self.costvec_val=[]

        #printing and such
        self.doplot = doplot
        self.doprint = doprint

    def fit(self, xtrain, ytrain, xval, yval, Niter, batch_size = 200, solver='gd'):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xval = xval
        self.yval = yval

        self.beta = np.random.randn(xtrain.shape[1],1)
        xaxis=[]
        indexes = np.arange(xtrain.shape[0])
        self.xtrain_tmp = xtrain
        self.ytrain_tmp = ytrain
        self.xval_tmp = xval
        self.yval_tmp = yval

        for iter in range(Niter):
            xaxis.append(iter+1)
            if solver == 'sgd':
                datapoints = np.random.choice(indexes, size=batch_size, replace=False)
                self.xtrain_tmp = self.xtrain[datapoints,:]
                self.ytrain_tmp = self.ytrain[datapoints]
                self.oneiteration()
                self.costs()
            elif solver == 'gd':
                self.oneiteration()
                self.costs()
            if self.doprint:
                print('Cost validation =', self.costvec_val[iter])
            if self.doplot:
                plt.plot(xaxis, self.costvec, 'b')
                plt.plot(xaxis, self.costvec_val, 'r')
                plt.pause(1e-12)
        plt.show()

        return self.costvec_val, self.costvec, xaxis

    def costs(self):
        cost_val = lrf.cost_log_ols(self.xval_tmp@self.beta,self.yval_tmp.T)
        self.costvec_val.append(cost_val.ravel())
        cost = lrf.cost_log_ols(self.xtrain_tmp@self.beta,self.ytrain_tmp.T)
        self.costvec.append(cost.ravel())

    def oneiteration(self):
        sig = lrf.sigmoid(self.xtrain_tmp@self.beta)
        gradients = lrf.gradient_ols(self.xtrain_tmp,self.ytrain_tmp,sig)
        self.beta -= self.eta*gradients

    def predict(self, x):
        predictions = x@self.beta
        sig = lrf.sigmoid(predictions)
        classes = np.round(sig)
        return classes
