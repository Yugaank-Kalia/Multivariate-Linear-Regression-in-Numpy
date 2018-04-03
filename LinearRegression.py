import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt

#load data
data = np.loadtxt('machine-learning-ex1/ex1/ex1data1.txt' , delimiter = ',')
x = np.c_[data[:,0]]
y = np.c_[data[:,1]]

def normalize(x):
    
    return (x - np.mean(x))/(np.amax(x)-np.amin(x))

x = normalize(x)
y = normalize(y)

#add coloumn of bias
X = np.c_[np.ones(data.shape[0]),x]

# m = number of training examples
# n = number of features
m = X.shape[0]
n = X.shape[1]

# plotting data using matplotlib.pyplot
def visualize_data(x,y):
    
    plt.plot(x,y,'o')
    plt.show()

visualize_data(x,y)

# iterations and alpha value initialization
iter = 10000
alpha = 0.003

# setting the weights to 0
theta = np.zeros((n,y.shape[1]))

# function to generate cost
def cost(theta,X,y):
    
    sqrErr = (np.dot(X,theta)-y)**2
    return np.sum(sqrErr)/(2*m)

def gradDescent(theta,X,y):
    
    hyp = np.dot(X,theta)
    cost_func = []
    
    for i in range(iter): 

        theta -= ((alpha/m)*(X.T.dot(hyp-y)))
        cost_func.append(cost(theta,X,y))

    return theta,cost_func

theta,cost_func = gradDescent(theta,X,y)

# plotting cost using matplotlib.pyplot
def visualize_cost(x,y):
    
    y_axis = np.array(cost_func)
    x_axis = np.array([i for i in range(iter)])
    plt.scatter(x_axis,y_axis)
    plt.show()

visualize_cost(x,y)

line = X.dot(theta)

def visualize_line(x,y):
    
    plt.plot(x,y,'x')
    plt.plot(x,line,'-')
    plt.show()

visualize_line(x,y)

# print(np.array(cost_func))

def normalEquation(X,y):
    return inv(X.T.dot(X)).dot(X.T).dot(y)

# theta = normalEquation(X,y)

def predict(weights, input):
    
    X = np.c_[np.ones(1), input] # a row of biases is added to the the input
    return (X.dot(weights))[0][0] # the weights computed using gradient descent are used to predict y

print("Prediction : ",predict(theta, 50)*1000)
