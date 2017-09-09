import csv
import cv2
import numpy as np
import os
import random
from scipy.misc import imread

def load_data(log, img_dir):
    images = []
    measurements = []
    with open(os.path.join(log)) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            # load center, left, right camera images
            corrections = [0.0, 0.25, -0.25]
            random_direction = random.choice([0, 1, 2])

            # for i in range(3):
            image_path = line[random_direction]
            filename = image_path.split('/')[-1]
            current_path = os.path.join(img_dir, filename)
            image = imread(current_path)
            images.append(image)
                # images.append(cv2.flip(image, 1))
            steering_center = float(line[3])
            steering_direction = steering_center + corrections[random_direction]
            # correction = 0.25
            # steering_left = steering_center + correction
            # steering_right = steering_center - correction
            measurements.append(steering_direction)
            # measurements.append(steering_center)
            # measurements.append(-steering_center)
            # measurements.append(steering_left)
            # measurements.append(-steering_left)
            # measurements.append(steering_right)
            # measurements.append(-steering_right)

    # augment data
    # augmented_images, augmented_measurements = [], []
    # for image, measurement in zip(images, measurements):
    #     augmented_images.append(image)
    #     augmented_measurements.append(measurement)
    #     augmented_images.append(cv2.flip(image, 1))
    #     augmented_measurements.append(measurement * -1.0)

    # x[:,:-1] = np.array([[9,8,7],[6,5,4]])
    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train
