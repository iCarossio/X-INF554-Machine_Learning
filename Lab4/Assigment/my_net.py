from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

###############################################################
#
# Important notes: 
# - Do not change any of the existing functions or parameter names, 
#       except in __init__, you may add/change parameter names/defaults values.
# - In __init__ set default values to the best ones, e.g., learning_rate=0.1
# - Training epochs/iterations should not be a parameter to __init__,
#   To train/test your network, we will call fit(...) until time (2 mins) runs out.
#
###############################################################

class Network():

    def __init__(self, learning_rate=0.001, num_hidden_layers=2, batch_size=32, weight_decay=0.0001, treshold=0.6, n_1=128, n_2=32):
        ''' initialize the classifier with default (best) parameters '''

        # Parameters
        self.num_epoch     = 0
        self.display_step  = 10
        self.treshold      = treshold
        self.batch_size    = batch_size
        self.weight_decay  = weight_decay
        self.learning_rate = learning_rate
        self.clock         = time()

        # Network Parameters
        n_input    = 294 # Data input (line shape: 294)
        n_hidden_1 = 0
        n_hidden_2 = 0
        n_hidden_3 = 0
        n_hidden_4 = 0
        n_hidden_5 = 0

        if num_hidden_layers == 5:
            n_hidden_1 = 256
            n_hidden_2 = 128
            n_hidden_3 = 64
            n_hidden_4 = 32
            n_hidden_5 = 16
            n_O_param  = n_hidden_5
        elif num_hidden_layers == 4:
            n_hidden_1 = 128
            n_hidden_2 = 64
            n_hidden_3 = 32
            n_hidden_4 = 16
            n_O_param  = n_hidden_4
        elif num_hidden_layers == 3:
            n_hidden_1 = 128
            n_hidden_2 = 64
            n_hidden_3 = 32
            n_O_param  = n_hidden_3
        else:
            n_hidden_1 = n_1
            n_hidden_2 = n_2
            n_O_param  = n_hidden_2

        n_classes  = 6   # Classes (0-5 digits)
 
        # tf Graph input
        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        W1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
        W2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
        W3 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]))
        W4 = tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4]))
        W5 = tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5]))        
        WO = tf.Variable(tf.random_normal([n_O_param, n_classes]))
        b1 = tf.Variable(tf.random_normal([n_hidden_1]))
        b2 = tf.Variable(tf.random_normal([n_hidden_2]))
        b3 = tf.Variable(tf.random_normal([n_hidden_3]))
        b4 = tf.Variable(tf.random_normal([n_hidden_4]))
        b5 = tf.Variable(tf.random_normal([n_hidden_5]))
        bO = tf.Variable(tf.random_normal([n_classes]))

        # Layers
        if num_hidden_layers == 5:
            layer1 = tf.nn.sigmoid(tf.matmul(self.x, W1) + b1)
            layer2 = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2)
            layer3 = tf.nn.sigmoid(tf.matmul(layer2, W3) + b3)
            layer4 = tf.nn.sigmoid(tf.matmul(layer3, W4) + b4)
            layer5 = tf.nn.sigmoid(tf.matmul(layer4, W5) + b5)
            self.pred = tf.add(tf.matmul(layer5, WO), bO)
        elif num_hidden_layers == 4:
            layer1 = tf.nn.sigmoid(tf.matmul(self.x, W1) + b1)
            layer2 = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2)
            layer3 = tf.nn.sigmoid(tf.matmul(layer2, W3) + b3)
            layer4 = tf.nn.sigmoid(tf.matmul(layer3, W4) + b4)
            self.pred = tf.add(tf.matmul(layer4, WO), bO)
        elif num_hidden_layers == 3:
            layer1 = tf.nn.sigmoid(tf.matmul(self.x, W1) + b1)
            layer2 = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2)
            layer3 = tf.nn.sigmoid(tf.matmul(layer2, W3) + b3)
            self.pred = tf.add(tf.matmul(layer3, WO), bO)        
        else:
            layer1 = tf.nn.sigmoid(tf.matmul(self.x, W1) + b1)
            layer2 = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2)
            self.pred = tf.add(tf.matmul(layer2, WO), bO)

        # Define loss and optimizer
        weight_decay_val = self.weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.cost        = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.pred)) + weight_decay_val
        self.opt         = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def fit(self,X,Y,warm_start=True,n_epochs=10):
        ''' train the network, and if warm_start, then do not reinit. the network
            (if it has already been initialized)
        '''

        batch_per_epoch = int(Y.shape[0]/self.batch_size)

        if not warm_start:
            print("No warm start. Session reset.")
            self.sess.close()
            self.__init__()

        # Training cycle
        for _ in range(n_epochs):
            self.num_epoch+=1
            avg_cost = 0.

            # Loop over all batches
            for i in range(batch_per_epoch):
                batch_x = X[self.batch_size*i:self.batch_size*(i+1),:]
                batch_y = Y[self.batch_size*i:self.batch_size*(i+1),:] 

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.opt, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})
                # Compute average loss
                avg_cost += c / batch_per_epoch

            # Display logs per epoch step
            if self.num_epoch % self.display_step == 0:
                print("Epoch: {:04d} cost={:.9f} time={:.4f}".format(self.num_epoch, avg_cost, time()-self.clock))

    def predict_proba(self,X):
        ''' return a matrix P where P[i,j] = P(Y[i,j]=1), 
        for all instances i, and labels j. '''
        predict_proba = self.sess.run(tf.nn.sigmoid(self.pred), feed_dict={self.x: X})

        # Attempted a normalisation
        #row_sums = predict_proba.sum(axis=1)
        #proba_norm = predict_proba / row_sums[:, np.newaxis]
        return predict_proba

    def predict(self,X):
        ''' return a matrix of predictions for X '''
        return (self.predict_proba(X) >= self.treshold).astype(int)
