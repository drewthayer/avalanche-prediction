import pickle
import sys
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD # stochastic gradient descent
import os

from scripts.modeling_scripts import print_scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # disable Tensorflow warnings

def define_hl_mlp_model(X, nn_hl, activ='sigmoid', loss='mean_squared_error', metrics=['mse']):
    """ Defines hidden layer mlp model
        X is the training array
        nn_hl is the desired number of neurons in the hidden layer
        activ is the activation function (could be 'tanh', 'relu', etc.)
    """
    num_coef = X.shape[1]
    model = Sequential() # sequential model is a linear stack of layers
    model.add(Dense(units=nn_hl,
                    input_shape=(num_coef,),
                    activation=activ,
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None))
    model.add(Dense(units=1,
                    activation=activ,
                    use_bias=True,
                    kernel_initializer='glorot_uniform'))
    sgd = SGD(lr=1.0, decay=1e-7, momentum=.9) # using stochastic gradient descent, default decay = L2
    model.compile(loss=loss, optimizer=sgd, metrics=metrics )
    return model

import matplotlib.pyplot as plt

if __name__=='__main__':
    zonename = sys.argv[1] # 'aspen' or 'nsj'
    case_select = sys.argv[2] # 'slab' or 'wet'

    # read engineered X,y data from pickle
    X_train, y_train, X_test, y_test = pickle.load(
    open('feature-matrices/{}_{}_matrices.pkl'.format(zonename, case_select),
    'rb'))

    # y to column vector
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    # set up neural network
    nn_hl = 4 # number of neurons in the hidden layer
    num_epochs = 30 # number of times to train on the entire training set
    batch_size = X_train.shape[0] # using batch gradient descent
    mlp = define_hl_mlp_model(X_train, nn_hl, loss='binary_crossentropy', metrics=['accuracy'])
    history = mlp.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, shuffle=True) #shuffle in between epochs
    y_pred = mlp.predict(X_test)

    print("\nTraining results: {}, {}".format(zonename, case_select))
    print("training accuracy = {:0.3f}".format(history.history['acc'][-1]))

    # map predicted probabilities as binary classification:
    threshold = 0.5
    y_hat = y_pred.copy()
    y_hat[y_hat >= threshold] = 1
    y_hat[y_hat < threshold] = 0

    # print test scores
    method_list = [accuracy_score, recall_score, precision_score]
    print_scores(y_test, y_hat, method_list)

    # plot training loss (get from history.history.keys())
    plt.plot(history.history['loss'])
    plt.ylabel('loss (binary cross-entropy)')
    plt.xlabel('epoch')
    plt.title('model training')
    plt.show()
