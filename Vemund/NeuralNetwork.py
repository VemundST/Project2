import numpy as np
import log_reg_functions as lrf
import functions as fx
from sklearn.metrics import f1_score

class ANN():
    '''
    ANN = Artificial Neural Network
    Functions belonging to ANN:
    - Initialization: where hyperparameters are defined, and what kind of network should be created (regression or classification)
    - add_layers: Adds user defined number of layers with user defined number of nodes:
    - Feed: Feed forward part of Network
    - back: Backpropogation
    - feed_out: Feed to output, useful for outputting test data.
    - costs: ouputs cost values per epochs
    - coefs: outputs final weights and biases
    - train: Trains the network, by employing self.feed and self.back
    '''
    def __init__(self, lmb=0, bias=0, eta=0.0001, early_stop_tol = 0.0,\
     early_stop_nochange=10, mode = 'classification', regularization = 'l2'):
        self.lmb=lmb
        self.bias = bias
        self.eta=eta
        self.regularization=regularization
        self.mode = mode
        self.early_stop_tol = early_stop_tol
        self.early_stop_nochange = early_stop_nochange

        self.n_layers=int()
        self.layers=dict()
        self.pred=dict()
        self.act=dict()

    def add_layers(self, n_neurons=[50,20], n_features=[20,1], n_layers=2):
        '''
        inputs:
        n_neurons = list of neurons
        n_features = list of features, n_features[i]=n_neurons[i+1]
        outputs:
        self.layers = dictionary containing layers and biases.

        PS: uses He Initialization for weights
        '''
        self.n_layers=n_layers
        for i in range(n_layers):
            if i == 0:
                layer_weights = np.random.randn(n_features[i], n_neurons[i])*np.sqrt(2/n_neurons[i])
            else:
                layer_weights = np.random.randn(n_features[i],\
                 n_neurons[i])*np.sqrt(2/n_neurons[i-1])
            self.layers['w'+str(i)] = layer_weights
            layer_bias = np.zeros(n_neurons[i]) + self.bias
            self.layers['b'+str(i)] = layer_bias

    def feed(self, design, activation=[lrf.sigmoid,lrf.sigmoid]):
        '''
        inputs:
        design = input design matrix
        activation = List containing activation functions

        outputs:
        self.act = dictionary containg the activation from the different layers
        '''
        for i in range(self.n_layers):

            if i==0:
                self.pred[str(i)] = np.matmul(design, self.layers['w'+str(i)]) + self.layers['b'+str(i)]
                self.act[str(i)] = activation[i](self.pred[str(i)])
            else:
                self.pred[str(i)] = np.matmul(self.act[str(i-1)],\
                 self.layers['w'+str(i)]) + self.layers['b'+str(i)]
                self.act[str(i)] = activation[i](self.pred[str(i)])

    def back(self, design, data, derivative=[lrf.sigmoid_deriv, lrf.sigmoid_deriv]):
        '''
        inputs:
        design = input design matrix
        data = target values

        outputs:
        self.layers = dictionary containing layers with updated weights and biases
        '''

        for i in np.arange(self.n_layers-1,-1,-1):


            if i==self.n_layers-1:
                if self.mode == 'regression':
                    error = (self.act[str(i)] - data)*derivative[i](self.act[str(i)])
                if self.mode == 'classification':
                    error = (self.act[str(i)] - data)*derivative[i](self.act[str(i)])
            else:
                error = np.matmul(error, self.layers['w'+str(i+1)].T)\
                 * derivative[i](self.act[str(i)])
            if i == 0:
                gradients_weights = (np.matmul(design.T, error))/len(data)
                gradients_bias = (np.sum(error, axis=0))/len(data)
            else:
                gradients_weights = (np.matmul(self.act[str(i-1)].T, error))/len(data)
                gradients_bias = (np.sum(error, axis=0))/len(data)

            if self.lmb>0.0:
                if self.regularization == 'l2':
                    gradients_weights += self.lmb * self.layers['w'+str(i)]

            self.layers['w'+str(i)] -= self.eta * gradients_weights
            self.layers['b'+str(i)] -= self.eta * gradients_bias

    def feed_out(self, design, activation=[lrf.sigmoid,lrf.sigmoid]):
        '''
        inputs:
        design = input design matrix
        activation =  activation functions

        outputs:
        self.act[self.act[str(self.n_layers-1)] = activation output from last layer
        '''
        for i in range(self.n_layers):

            if i==0:
                self.pred[str(i)] = np.matmul(design, self.layers['w'+str(i)])\
                 + self.layers['b'+str(i)]
                self.act[str(i)] = activation[i](self.pred[str(i)])
            else:
                self.pred[str(i)] = np.matmul(self.act[str(i-1)],\
                 self.layers['w'+str(i)]) + self.layers['b'+str(i)]
                self.act[str(i)] = activation[i](self.pred[str(i)])
        return self.act[str(self.n_layers-1)]

    def costs(self):
        '''
        Outputs the train and test loss as a function of epoch.
        '''
        return self.cost_val, self.cost_train

    def coefs(self):
        '''
        Outputs the trained coefficients.
        '''
        return self.layers

    def train(self, epochs, batch_size, x, y, activation, derivative,\
     xvalidation, yvalidation, verbose=False):

        '''
        inputs:
        epochs = max epochs
        batch_size = self explanatory
        x,y = the datset used for training
        activation = list of activation functions
        derivative = list of derivatives (i know this can be done better)
        xvalidation = validation design matrix used in early stopping
        yvalidation = validation output data used in early stopping
        verbose == False no printing of validation loss, == True validation loss is printed each epoch

        outputs:
        No outputs, but hopefully a well trained network 
        '''
        tmp=int(len(y)/batch_size)
        Niter = min(200,tmp)
        indexes = np.arange(len(y))
        cost = np.empty([epochs])

        self.cost_val = list()
        self.cost_train = list()

        for i in range(epochs):
            for j in range(Niter):


                datapoints = np.random.choice(indexes, size=batch_size, replace=False)
                batch_x = x[datapoints,:]
                batch_y = y[datapoints]

                self.feed(batch_x, activation)
                self.back(batch_x,batch_y, derivative)

            pred_val = self.feed_out(xvalidation, activation)
            pred_train = self.feed_out(batch_x, activation)
            if self.mode == 'regression':
                self.cost_val.append(fx.MSE(pred_val.ravel(),yvalidation.ravel()))
                self.cost_train.append(fx.MSE(pred_train.ravel(),batch_y.ravel()))
            if self.mode == 'classification':
                self.cost_val.append(lrf.cost_log_ols(pred_val.ravel(),yvalidation.T))
                self.cost_train.append(lrf.cost_log_ols(pred_train.ravel(),batch_y.T))
            if i > self.early_stop_nochange:
                avg_indx_full = np.arange(i-self.early_stop_nochange,i)
                avg_indx_full.astype(int)
                avg_indx = np.arange(i-5,i)
                avg_indx.astype(int)

                if -self.early_stop_tol<np.mean(np.array(self.cost_val)[avg_indx]) - np.mean(np.array(self.cost_val)[avg_indx_full]):
                    break

            if verbose:
                print('Epoch', i+1, 'loss', self.cost_val[i] )
