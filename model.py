from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Cropping2D
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Lambda
from keras.layers import ELU
from keras.models import Sequential

from argparse import ArgumentParser
from sklearn.utils import shuffle

from loader import data

parser = ArgumentParser(description='Train an autonomous vehicle model')
parser.add_argument('-lr', type=float, default=None)
parser.add_argument('-i', action="store", dest="images")
parser.add_argument('-l', action="store", dest="log")
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()
print(args)

X_train, y_train = data.load_data(args.log, args.images)

# When computing validation split with Keras, the validation dataset
# is the last X percent of the data. There is no shuffling.
X_train, y_train = shuffle(X_train, y_train)

input_shape = X_train.shape[1:]
N = X_train.shape[0]

print("Loaded %d samples of shape %s" % (N, input_shape))

epochs = args.epochs

# Use the comma.ai model
model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
model.add(Conv2D(16, 8, 8, subsample = (4, 4), border_mode = "same"))
model.add(ELU())
model.add(Conv2D(32, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(ELU())
model.add(Conv2D(64, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=args.batch_size, validation_split=0.2, shuffle=True, nb_epoch=epochs)
model.save('model.h5')