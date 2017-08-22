from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential

from argparse import ArgumentParser

from loader import data

parser = ArgumentParser(description='Train an autonomous vehicle model')
parser.add_argument('-lr', type=float, default=None)
parser.add_argument('-i', action="store", dest="images")
parser.add_argument('-l', action="store", dest="log")
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--droprate', type=float, default=0.5)
parser.add_argument('--activation', action="store", dest="activation", default='relu')

args = parser.parse_args()
print(args)

X_train, y_train = data.load_data(args.log, args.images)

input_shape = X_train.shape[1:]
N = X_train.shape[0]

print("Loaded %d samples of shape %s" % (N, input_shape))

epochs = args.epochs
learning_rate = args.lr
droprate = args.droprate
activation = args.activation
conv1_nfilters = 12
conv2_nfilters = 32
fc1_nodes = 120
fc2_nodes = 84


model = Sequential()
model.add(Conv2D(filters=conv1_nfilters, kernel_size=(5, 5), input_shape=input_shape))
model.add(Activation(activation))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(droprate))
model.add(Conv2D(filters=conv2_nfilters, kernel_size=(5, 5)))
model.add(Activation(activation))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(droprate))
model.add(Flatten())
model.add(Dense(fc1_nodes))
model.add(Activation(activation))
model.add(Dropout(droprate))
model.add(Dense(fc2_nodes))
model.add(Activation(activation))
model.add(Dropout(droprate))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
model.save('model.h5')