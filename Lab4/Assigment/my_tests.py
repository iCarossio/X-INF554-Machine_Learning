#!/usr/bin/env python
# coding: utf-8

### This has been used for testing optimal parameters ###

# # Network definition

# In[1]:


from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# In[2]:


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


# In[3]:


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
        row_sums = predict_proba.sum(axis=1)
        proba_norm = predict_proba / row_sums[:, np.newaxis]
        return predict_proba

    def predict(self,X):
        ''' return a matrix of predictions for X '''
        return (self.predict_proba(X) >= self.treshold).astype(int)


# # Main test

# In[4]:


# Test our classifier 
h = Network()

t0 = time()
while (time() - t0) < 120:
    h.fit(X_train,Y_train)

print("Trained %d epochs in %d seconds." % (h.num_epoch,int(time() - t0)))


# In[5]:


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


# In[6]:


results(X_test)


# # First test

# ## For nb_layers and learning_rate

# In[4]:


def run_network_1(nb_layers=2, learning_rate=0.001):
    
    print("\n\n########## Training NN with nb_layers={}, learning_rate={} ##########".format(nb_layers,learning_rate))

    t0 = time()
    h = Network(learning_rate=learning_rate, num_hidden_layers=nb_layers)

    t0 = time()
    while (time() - t0) < 120:
        h.fit(X_train,Y_train)

    print("Trained %d epochs in %d seconds." % (h.num_epoch,int(time() - t0)))

    proba = h.predict_proba(X_test)
    Y_pred = (proba >= 0.5).astype(int)
    print("sum(sum(Y_pred)) = {}".format(sum(sum(Y_pred))))
    print("sum(sum(Y_test)) = {}".format(sum(sum(Y_test))))
    tester = Y_pred != Y_test
    loss = np.mean(tester)
    print("Hamming loss     =", loss)

    return h.num_epoch, loss

nb_layers      = [2,3,4,5]
learning_rates = [0.0001, 0.001, 0.01, 0.1]

epoc_layers    = []
epoc_learning  = []

loss_layers    = []
loss_learning  = []
# In[6]:


losses = dict()


# In[7]:


for i in nb_layers:
    for j in learning_rates:
        key = "{}+{}".format(i,j)
        epoch, loss = run_network_1(nb_layers=i, learning_rate=j)
        losses[key]=loss


# ## For batch_size and weight_decay

# In[4]:


def run_network_2(batch_size, weight_decay):
    
    print("\n\n########## Training NN with batch_size={}, weight_decay={} ##########".format(batch_size,weight_decay))

    t0 = time()
    h = Network(batch_size=batch_size, weight_decay=weight_decay)

    t0 = time()
    while (time() - t0) < 120:
        h.fit(X_train,Y_train)

    print("Trained %d epochs in %d seconds." % (h.num_epoch,int(time() - t0)))

    proba = h.predict_proba(X_test)
    Y_pred = (proba >= 0.5).astype(int)
    print("sum(sum(Y_pred)) = {}".format(sum(sum(Y_pred))))
    print("sum(sum(Y_test)) = {}".format(sum(sum(Y_test))))
    tester = Y_pred != Y_test
    loss = np.mean(tester)
    print("Hamming loss     =", loss)

    return h.num_epoch, loss


# In[5]:


batch_sizes    = [8,16,32,64]
weight_decays  = [0,0.00001, 0.0001, 0.001]
losses_2 = dict()

for i in batch_sizes:
    for j in weight_decays:
        key = "{}+{}".format(i,j)
        epoch, loss = run_network_2(batch_size=i, weight_decay=j)
        losses_2[key]=loss


# In[6]:


losses_2


# ## For n_1 and n_2

# In[6]:


def run_network_3(n_1, n_2):
    
    print("\n\n########## Training NN with n_1={}, n_2={} ##########".format(n_1,n_2))

    t0 = time()
    h = Network(n_1=n_1, n_2=n_2)

    t0 = time()
    while (time() - t0) < 120:
        h.fit(X_train,Y_train)

    print("Trained %d epochs in %d seconds." % (h.num_epoch,int(time() - t0)))

    proba = h.predict_proba(X_test)
    Y_pred = (proba >= 0.5).astype(int)
    print("sum(sum(Y_pred)) = {}".format(sum(sum(Y_pred))))
    print("sum(sum(Y_test)) = {}".format(sum(sum(Y_test))))
    tester = Y_pred != Y_test
    loss = np.mean(tester)
    print("Hamming loss     =", loss)

    return h.num_epoch, loss


# In[5]:


values = [16,32,64,128,256]
losses_3 = dict()

for a in range(len(values)):
    for b in range(a+1):
        i = values[a]
        j = values[b]
        key = "{}+{}".format(i,j)
        epoch, loss = run_network_3(n_1=i, n_2=j)
        losses_3[key]=loss


# In[12]:


sortedkeys = sorted(losses_3, key=str.lower)
for i in sortedkeys:
    print(i,losses_3[i])


# ## For treshold

# In[7]:


proba = h.predict_proba(X_test)


# In[8]:


row_sums = proba.sum(axis=1)
proba_norm = proba / row_sums[:, np.newaxis]


# In[9]:


def results_t(treshold=0.5):
    Y_pred = (proba >= treshold).astype(int)
    Y_pred_norm = (proba_norm >= treshold).astype(int)
    loss = np.mean(Y_pred != Y_test)
    loss_norm = np.mean(Y_pred_norm != Y_test)
    return loss, loss_norm


# In[10]:


tresholds = np.linspace(0,1,101)
losses_4 = []
losses_4_norm = []


# In[11]:


for t in tresholds:
    loss, loss_norm = results_t(treshold=t)
    losses_4.append(loss)
    losses_4_norm.append(loss_norm)


# In[12]:


print(losses_4.index(min(losses_4)), min(losses_4))
print(losses_4_norm.index(min(losses_4)), min(losses_4_norm))


# In[13]:


plt.figure()
plt.plot(tresholds, losses_4, 'r-', label="Without normalization")
plt.plot(tresholds, losses_4_norm, 'b-', label="With normalization")
plt.xlabel('Treshold')
plt.ylabel("Hamming Loss")
plt.legend()
plt.title("Test of tresholds")
plt.show()


# # Execution time

# In[18]:


t0 = time()
Y_pred = h.predict(X_test)
t1 = time() - t0
loss = np.mean(Y_pred != Y_test)
print(t1)