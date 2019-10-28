


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
