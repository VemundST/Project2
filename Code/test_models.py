import numpy as np
import NeuralNetwork as nn
import log_reg_functions as lrf

def test_models(xtrain,ytrain,xval,yval,n_features, n_neurons, n_layers,activation, derivation, batch_size,etavec, lmbvec, bias=0,epochs=1000, early_stop_tol = 0, early_stop_nochange=50, mode = 'classification', regularization = 'l1'):

    best_net = object()
    train_accuracy = np.ones((len(etavec), len(lmbvec)))*1000
    for i, etas in enumerate(etavec):
        for j, lmb in enumerate(lmbvec):

            np.random.seed(2019)
            neural_net = nn.ANN(lmb=lmb, bias=bias, eta=etas, early_stop_tol, early_stop_nochange, mode, regularization)
            neural_net.add_layers(n_features, n_neurons , n_layers)
            neural_net.train(epochs, batch_size, xtrain,ytrain,activation,derivative, xval, yval, verbose=False)

            pred = neural_net.feed_out(xval, activation)
            val_accuracy[i,j] =  lrf.cost_log_ols(pred.ravel(),yval.T)
            if val_accuracy[i,j]<np.min(val_accuracy):
                best_net = neural_net
            print('Validation loss for', '\u03B7 =', etas, '&', '\u03BB =', lmb, '=', val_accuracy [i,j])
    return best_net, val_accuracy
