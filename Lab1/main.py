from numpy import *
from matplotlib.pyplot import *
from scipy.linalg import inv

# Load the data

data = loadtxt('data/data_train.csv', delimiter=',')

# Prepare the data

X = data[:,0:-1]
y = data[:,-1]

# Inspect the data

figure()
hist(X[:,1], 10)

# <TASK 1>

figure()
plot(X[:,1],X[:,2], 'o')
xlabel('x2')
ylabel('x3')

# <TASK 2>

show()

# Standardization

# <TASK 2>

# Feature creation

from tools import poly_exp
Z = poly_exp(X,2)

Z = column_stack([ones(len(Z)), Z])

# Building a model

# <TASK 3>

# Evaluation 

#y_pred = dot(Z_test,w)

# <TASK 4>

from tools import MSE

# <TASK 5>
# <TASK 6>
# <TASK 7>

# <TASK 8: You will need to make changes from '# Feature creation'
#          To get the exact results, you will need to reverse the second part of Task 7 (your own modifications)>
