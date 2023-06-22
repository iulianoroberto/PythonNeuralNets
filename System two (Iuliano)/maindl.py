#
#       Main program to train / validate / test the models
#
#       Software is distributed without any warranty;
#       bugs should be reported to antonio.pecchia@unisannio.it
#


from utils import  transformData, evaluatePerformance
from FeedforwardNN import FeedforwardNN
from AutoEncoder import AutoEncoder

# Fare attenzione al file di training selezionato quando si usa autoencoder
train = r'.\Data\TRAIN_AE.csv'
validation = r'.\Data\VALIDATION.csv'
test = r'.\Data\TEST.csv'

# Operazione di codifica
x_train, y_train, l_train, x_val, y_val, l_val, x_test, y_test, l_test = transformData( train, validation, test )

print( "x_train = " + str(x_train.shape) )
print( "y_train = " + str(y_train.shape) )

print( "x_train = " + str(x_val.shape) )
print( "y_train = " + str(y_val.shape) )

input_dim = x_train.shape[1]

# Istanziamo il modello
#ffnn = FeedforwardNN( input_dim = input_dim)
# Facciamo il summary del modello
#ffnn.summary()
# Addestriamo il modello
#ffnn.train(x_train, y_train)

#outcome = ffnn.predict(x_test)
#evaluatePerformance(outcome, l_test)

ae = AutoEncoder(input_dim=input_dim)
ae.summary()
# Operazione che potrebbe risultare anomala ma cosi lo forziamo a ricostruire i livelli
ae.train(x_train, x_train)

# Giocare con il validation set e facendo tuning dei parametri
outcome = ae.predict(x_test)
evaluatePerformance(outcome, l_test)
ae.plot_reconstruction_error(x_test, l_test)
