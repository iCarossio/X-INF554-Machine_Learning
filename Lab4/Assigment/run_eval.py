from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from my_net import Network

###############################################################
# This is an example script, you may modify it as you wish
###############################################################

# Parameters
L = 6

# Load and parse the data (N instances, D features, L labels)
data = pd.read_csv('data/scene.csv') # Load data from CSV with Pandas
XY = data.values
N,DL = XY.shape
D = DL - L
Y = XY[:,0:L].astype(int)
X = XY[:,L:D+L]

# Split into train/test sets
n = int(N*6/10)
X_train = X[0:n]
Y_train = Y[0:n]
X_test = X[n:]
Y_test = Y[n:]

# Get class names
class_names = list(data.columns.values[:L])

# Test our classifier 
h = Network()
t0 = time()
while (time() - t0) < 120:
    h.fit(X_train,Y_train)
print("Trained %d epochs in %d seconds." % (h.num_epoch,int(time() - t0)))

def results(X_test):
    proba = h.predict_proba(X_test)
    Y_pred = h.predict(X_test)
    print("sum(sum(Y_pred)) = {}".format(sum(sum(Y_pred))))
    print("sum(sum(Y_test)) = {}".format(sum(sum(Y_test))))
    tester = Y_pred != Y_test
    loss = np.mean(tester)
    print("Hamming loss     =", loss)

    for i in range(len(tester)):
        if any(tester[i]):
            formattedProba = [ '%.2f' % elem for elem in proba[i] ]
            print("{:03d} : PROB = {} | PRED = {}  |  TEST = {}".format(i, formattedProba, Y_pred[i],Y_test[i]))

results(X_test)