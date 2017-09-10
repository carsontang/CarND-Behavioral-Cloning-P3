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

import csv
import numpy as np
import os
from scipy.misc import imread

def load_data(log, img_dir):
    images = []
    measurements = []
    with open(os.path.join(log)) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            # load center, left, right camera images
            for i in range(3):
                image_path = line[i]
                filename = image_path.split('/')[-1]
                current_path = os.path.join(img_dir, filename)
                image = imread(current_path)
                images.append(image)
            steering_center = float(line[3])
            correction = 0.25
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            measurements.append(steering_center)
            measurements.append(steering_left)
            measurements.append(steering_right)

    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train

parser = ArgumentParser(description='Train an autonomous vehicle model')
parser.add_argument('-i', action="store", dest="images", default="/home/carson/Documents/CarND-Behavioral-Cloning-P3/data/IMG")
parser.add_argument('-l', action="store", dest="log", default="/home/carson/Documents/CarND-Behavioral-Cloning-P3/data/driving_log.csv")
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()
print(args)

X_train, y_train = load_data(args.log, args.images)

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
model.summary()
model.fit(X_train, y_train, batch_size=args.batch_size, validation_split=0.2, shuffle=True, nb_epoch=epochs)
model.save('model.h5')