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
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = os.path.join(img_dir, filename)
            image = cv2.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train
