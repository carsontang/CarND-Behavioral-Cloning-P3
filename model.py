from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential

from loader import data

X_train, y_train = data.load_data('/Users/ctang/dev/CarND-Behavioral-Cloning-P3')

input_shape = X_train.shape[1:]
N = X_train.shape[0]

print("Loaded %d samples of shape %s" % (N, input_shape))

epochs = 60
droprate = 0.5
conv1_nfilters = 12
conv2_nfilters = 32
fc1_nodes = 120
fc2_nodes = 84
learning_rate = 0.001
activation = 'relu'


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