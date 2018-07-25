import cv2
import numpy as np
import pandas as pd
import pickle
import os
import scipy

from itertools import islice
import matplotlib.pyplot as plt

LIMIT = None

DATA_FOLDER = 'dataset'
TRAIN_FILE = os.path.join(DATA_FOLDER, 'data.txt')

def preprocess(img):
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))
    return resized

def return_data():
    X = []
    y = []
    features = []

    with open(TRAIN_FILE) as fp:
        for line in islice(fp, LIMIT):
            path = line.strip().split()[0]
            angle = line.strip().split()[1].split(",")[0]

            full_path = os.path.join(DATA_FOLDER + "/data/", path)

            X.append(full_path)
            y.append(float(angle)*scipy.pi/180)

    for i in range(len(X)):
        img = plt.imread(X[i])
        features.append(preprocess(img))

    features = np.array(features).astype(np.float32)
    labels = np.array(y).astype(np.float32)

    with open("features", "wb") as f:
        pickle.dump(features, f, protocol=4)
    with open("labels", "wb") as f:
        pickle.dump(labels, f, protocol=4)

return_data()
