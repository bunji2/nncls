import numpy as np
import urllib.request
import os

IRIS_TRAINING = 'iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = 'iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

if not os.path.exists(IRIS_TRAINING):
    with open(IRIS_TRAINING, 'wb') as f:
        f.write(urllib.request.urlopen(IRIS_TRAINING_URL).read())

if not os.path.exists(IRIS_TEST):
    with open(IRIS_TEST, 'wb') as f:
        f.write(urllib.request.urlopen(IRIS_TEST_URL).read())

training_data = np.loadtxt(IRIS_TRAINING, delimiter=',', skiprows=1)
#print(training_data)
train_x = training_data[:, :-1]
train_y = training_data[:, -1]

test_data = np.loadtxt(IRIS_TEST, delimiter=',', skiprows=1)
test_x = test_data[:, :-1]
test_y = test_data[:, -1]


#print(train_x[0]) # ===> [6.4 2.8 5.6 2.2]
#print(train_y[0]) # ===> 2.0