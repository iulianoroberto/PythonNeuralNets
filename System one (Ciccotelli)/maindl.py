#
#       Main program to train / validate / test the models
#
#       Software is distributed without any warranty;
#       bugs should be reported to antonio.pecchia@unisannio.it
#

import tensorflow
import keras

from utils import transformData, evaluatePerformance
from FeedforwardNN import FeedforwardNN
from AutoEncoder import AutoEncoder

train = "./Data/TRAIN_AE.csv"
validation = './Data/VALIDATION.csv'
test = './Data/TEST.csv'

x_train, y_train, l_train, x_validation, y_validation, l_validation, x_test, y_test, l_test = transformData(train, validation, test)

print("x_train = " + str(x_train.shape))
print("y_train = " + str(y_train.shape))

print("x_validation = " + str(x_validation.shape))
print("y_validation = " + str(y_validation.shape))

input_dim = x_train.shape[1]

#ffnn = FeedforwardNN( input_dim = input_dim )
#ffnn.summary()
# Abbiamo instnziato il modello, ne abbiamo fatto un summary, ora procediamo ad addestrarlo:
#ffnn.train(x_train, y_train)
#outcome = ffnn.predict(x_test)
# Confrontiamo l'uscita del modello con i valori veri e propri (l_validation)
#evaluatePerformance(outcome, l_test)

ae = AutoEncoder(input_dim = input_dim)
ae.summary()
ae.train(x_train, x_train)
outcome = ae.predict(x_test)
evaluatePerformance(outcome, l_test)
ae.plot_reconstruction_error(x_test, l_test)

print(tensorflow.__version__, keras.__version__)
