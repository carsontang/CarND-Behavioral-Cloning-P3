import csv
import cv2
import numpy as np
import os

def load_data(log, img_dir):
    images = []
    measurements = []
    with open(os.path.join(log)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # load center, left, right camera images
            for i in range(3):
                image_path = line[i]
                filename = image_path.split('/')[-1]
                current_path = os.path.join(img_dir, filename)
                image = cv2.imread(current_path)
                images.append(image)
            steering_center = float(line[3])
            correction = 0.2
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            measurements.append(steering_center)
            measurements.append(steering_left)
            measurements.append(steering_right)
    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train
