#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *
import matplotlib.pyplot as plt
import scipy.optimize as op

def sigmoid(z):
    # Computes the sigmoid of z.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the sigmoid function.
    
    return 1/(1+exp(-z))

def proba(b,w1,w2,x1,x2):
    return sigmoid(b+ x1*w1 + x2*w2)


def cost(theta, X, y): 
    # Computes the cost using theta as the parameters for logistic regression. 
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Calculate the error J of the decision boundary
    #               that is described by theta (see the assignment 
    #               for more details).
    
    # theta[0] = b
    # theta[1] = w1
    # theta[2] = w2
    b  = theta[0]
    w1 = theta[1]
    w2 = theta[2]
    N  = X.shape[0]
    c  = y
    
    res = 0
    
    for i in range(0,N):
        res += -1/N*( c[i]*log(proba(b,w1,w2,X[i][0],X[i][1])) + (1-c[i])*log(1-proba(b,w1,w2,X[i][0],X[i][1])) )
    
    return res
    


def compute_grad(theta, X, y):
    # Computes the gradient of the cost with respect to
    # the parameters.
    
    grad = zeros(size(theta)) # initialize gradient
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of cost for each theta.
    
    
    b  = theta[0]
    w1 = theta[1]
    w2 = theta[2]
    N  = X.shape[0]
    c  = y
    
    grad = 0
    
    for i in range(0,N):
        grad += 1/N*(proba(b,w1,w2,X[i][0],X[i][1]) - c[i]) * X[i]
    
    return grad




def predict(theta, X):
    # Predict whether each label is 0 or 1 using learned logistic 
    # regression parameters theta. The threshold is set at 0.5
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Predict the label of each instance of the
    #               training set.
    
    b  = theta[0]
    w1 = theta[1]
    w2 = theta[2]
    N  = X.shape[0]
    res = {}

    for i in range (1,N):
        res[i] = sigmoid(proba(b,w1,w2,X[i][0],X[i][1]))
    
    return res
    
    # =============================================================
    


#======================================================================
# Load the dataset
# The first two columns contains the exam scores and the third column
# contains the label.
data = loadtxt('./data/data.txt', delimiter=',')
 
X = data[:, 0:2]
y = data[:, 2]

# Plot data 
pos = where(y == 1) # instances of class 1
neg = where(y == 0) # instances of class 0
plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not Admitted'])
plt.show()


#Add intercept term to X
X_new = ones((X.shape[0], 3))
X_new[:, 1:3] = X
X = X_new

# Initialize fitting parameters
initial_theta = random.randn(3,1)

# Run minimize() to obtain the optimal theta
Result = op.minimize(fun = cost, x0 = initial_theta, args = (X, y), method = 'TNC',jac = compute_grad);
theta = Result.x;

# Plot the decision boundary
plot_x = array([min(X[:, 1]) - 2, max(X[:, 2]) + 2])
plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
plt.plot(plot_x, plot_y)
plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='b')
plt.scatter(X[neg, 1], X[neg, 2], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Decision Boundary', 'Admitted', 'Not Admitted'])
plt.show()

# Compute accuracy on the training set
p = predict(array(theta), X)
# Evaluation
accuracy = mean(p == y)
print("\nAccuracy: %4.3f" % accuracy)


# In[ ]:




