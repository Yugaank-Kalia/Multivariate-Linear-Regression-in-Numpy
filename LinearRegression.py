import numpy as np 
from numpy.linalg import inv

#load data
data = np.loadtxt('machine-learning-ex1-2/ex1/ex1data1.txt' , delimiter = ',')
x = data[:,0]
y = np.c_[data[:,1]]

#add coloumn of bias
X = np.c_[np.ones(data.shape[0]),x]

# m = number of training examples
# n = number of features
m = X.shape[0]
n = X.shape[1]

# iterations and alpha value initialization
iter = 1500
alpha = 0.03

# setting the weights to 0
theta = np.zeros((n,y.shape[1]))

# function to generate cost
def cost(theta,X,y):
    sqrErr = (np.dot(X,theta)-y)**2
    return (1./2*m)*np.sum(sqrErr)

def feature_normalize(X):
    global n
    means = np.array([np.mean(X[:,i]) for i in range(n)])
    std = np.array([np.std(X[:,i]) for i in range(n)])
    normalize = (X-means)/std
    
    return np.c_[normalize[:,1]]

def gradDescent(theta,X,y):
    global iter,alpha,m
    for i in range(iter):
        gradient = (1./m)*np.dot(X.T,np.dot(X,theta)-y)
        theta = theta - alpha*gradient

    return theta 

def normalEquation(X,y):
    return inv(X.T.dot(X)).dot(X.T).dot(y)

print("Gradient Descent : ",gradDescent(theta,X,y),"\n\n\n\nNormal Equation : ",normalEquation(X,y))