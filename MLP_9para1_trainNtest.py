'''Train a simple deep NN on the MNIST dataset.

We will train with 9 digits and then test with all the digits.
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, MaskedLayer
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras.utils import np_utils
from keras.utils.visualize_util import plot
import keras.backend as K
import pylab as p
import matplotlib.pyplot as plt
#import seaborn

np.random.seed(1337)  # for reproducibility
batch_size = 128
nb_classes = 9
nb_epoch = 10
digito = 9

# Dropout para utilização na fase de teste
class MyDropout(MaskedLayer):
    def __init__(self, p, **kwargs):
        super(MyDropout, self).__init__(**kwargs)
        self.p = p

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.p > 0.:
            X = K.dropout(X, level=self.p)
        return X

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'p': self.p}
        base_config = super(MyDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# separa os indices para teste e treinamento
indices_teste = np.array([i for i,c in enumerate (y_test) if c == digito])
indices = np.array([i for i,c in enumerate (y_train) if c == digito])

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = np.delete(X_train, indices, axis = 0)
X_test_aux = X_test
X_test = np.delete(X_test, indices_teste,axis = 0)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# np.delete - retira do set de dados y_tain os indices e axis = 0 indica linhas completas
y_train = np.delete(y_train, indices, axis = 0)
# np_utils.to_categorical transforma um numero em uma mascaras de 0 e 1 para comparar com a saida
Y_train = np_utils.to_categorical(y_train, nb_classes)
y_test_aux = y_test
y_test = np.delete(y_test, indices_teste, axis = 0)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Camada de entrada
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(MyDropout(0.5))

#Camanda Intermediaria
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(MyDropout(0.5))
#Camada de saida
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
#ver sintaxe correta para gravar arquivo png do modelo
#model.save_weights('MLP_9_2_1.h5')
#model.load_weights('MPL_9_2_1.h5')
T = 100

def evaluate(X, l=2, p=0.5):
#    N = X_train.shape[0]
    probs = []
    for i in range(T):
        probs.append(model.predict(np.array(X)))
    pred_mean = np.mean(probs, axis=0)
    pred_std = np.std(probs, axis=0)
    #tau = l**2 * (1 - p) / (2 * N * rg.l2)
    #pred_variance += tau**-1
    return pred_mean, pred_std

def plot_class(cl):
    indexes = [i for i, c in enumerate(y_test_aux) if c != cl]
    m, s = evaluate(np.delete(X_test_aux, indexes, axis=0))
#    aux = np.delete(X_test_aux, indexes, axis=0)
#    aux = aux.shape[0]
#    m = m/aux
#    s = s/aux
    plt.figure('All_Outputs_SD' + str(cl))
    plt.title('All Outputs SD #' + str(cl))
    plt.hist(s)
    plt.xlim(0, 0.4)
    plt.xlabel('Standard Deviation')
    plt.figure('All_Outputs_Mean' + str(cl))
    plt.title('All Outputs Mean #' + str(cl))
    plt.hist(m)
    plt.xlim(0, 1)
    plt.xlabel('Mean')
    
for i in range(10):
    plot_class(i)
