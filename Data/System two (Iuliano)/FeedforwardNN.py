#
#       Feedforward neural network for binary classification -
#
#
#       Software is distributed without any warranty;
#       bugs should be reported to antonio.pecchia@unisannio.it
#

import numpy as np
import pandas as pd
import matplotlib as mpl
import tensorflow as tf
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras import regularizers, initializers

# Ha fissato il seed del generatore di numeri casuali
tf.random.set_seed(123)

class FeedforwardNN():

    def __init__(self, input_dim):

        # 81
        input_layer = Input ( shape=(input_dim,))

        # Invece della tanh potremmo usare relu
        layer = Dense (60, activation='tanh', kernel_initializer=initializers.RandomNormal ()) (input_layer)
        layer = Dense (24, activation='relu', kernel_initializer=initializers.RandomNormal ()) (layer)
        # Dense sta per densamente connesso
        # Valore inizale dei pesi dei neuroni assegnata con kernel che deve essere randomica
        layer = Dense (24, activation='tanh', kernel_initializer=initializers.RandomNormal ()) (layer)
        layer = Dense (2, activation='relu', kernel_initializer=initializers.RandomNormal ()) (layer)

        # Ora dobbiamo dire a quale layer precendente lo vogliamo collegare, questo lo facciamo specificando il layer con le parentesi
        # L'utlimo livello lo lasciamo softmax
        output_layer = Activation ( activation='softmax') (layer)

        self.classifier = Model ( inputs = input_layer, outputs = output_layer)

    def summary(self, ):
        self.classifier.summary()

    def train(self, x, y):

        epochs = 90
        batch_size = 1024
        validation_split = 0.1  # Lo 0.1% è riservato per monitorare la perdita

        self.classifier.compile( optimizer='rmsprop', loss='categorical_crossentropy')

        # Shuffle=True prende gli esempi dal dataset in modo random e serve e evitare la polarizzazione in fase di training
        history = self.classifier.fit( x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=True, verbose=2)

        # -----------------------------------------------#
        #           instructor-provided code            #
        # -----------------------------------------------#
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.show()

        df_history = pd.DataFrame(history.history)
        return df_history

    # Preso in ingresso un numero di punti (arbitrario) e ci dice qual'è l'uscita prevista dalla rete
    def predict ( self, x_evaluation ):

        predictions = self.classifier.predict(x_evaluation)

        outcome = predictions[:, 0] > predictions[:, 1]
        return outcome
