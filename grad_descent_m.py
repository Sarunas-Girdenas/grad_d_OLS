"""This is the simple implementation of a batch gradient descent algorithm.
Partially inspired by https://gist.github.com/marcelcaraciolo/1321575
This is extension for multivariate OLS
Written by Sarunas Girdenas, sg325@exeter.ac.uk, July, 2014"""

import numpy as np
import matplotlib.pyplot as plt

# Import some data

Data = np.genfromtxt('some_data_file.txt',delimiter=',',dtype=None)

# Name the variables and assign column to each of them

for i in range(0,Data.shape[1]):
	globals()['Data_Var_{}'.format(i)] = Data[0:len(Data),i]

thetas = Data.shape[1] # number of variables on right hand side + intercept

# notice that Data_Var_0 is the first column of your data
# So now our hypothesis is 
# y = alpha_0 +alpha_1*X1 + alpha_2*X2 + ...

# clear redundand Data

del Data

# Let's do gradient descent now

g = Data_Var_0.size # size of the sample

theta = np.zeros([thetas,1]) # container for theta

# Initialize theta values

# Initial guess for thetas

theta[0][0] = 5
theta[1][0] = 2
theta[2][0] = 0.3  

alpha = 0.01 # learning rate (step of algorithm)

# set the hypothesis, in our case it is:
# y = theta_0 + theta_1*X1 + theta_2*X2

X = np.asarray([Data_Var_1, Data_Var_2])
y = np.asarray([Data_Var_0])

# Add column of 1 to X or an intercept

X = np.concatenate((np.ones([g,1]),X.T),axis=1)

# Define cost function

def cost_f(X,y,theta):
 	training   = y.shape[1]
 	pred       = X.dot(theta).flatten()
 	sq_err     = (pred - y) ** 2
 	J = (1.0 / (2*training)) * sq_err.sum()
 	return J

m = g # sample size

max_iter = 10000 # No of maximum iterations
no_iter  = 0 # initialize 
Er       = 0.0000000000000000000000000001 # tolerance for error

# Create empty matrix to store thetas

theta_vals = np.zeros([max_iter,thetas])

# first column of theta_vals is theta_0, second one is theta_1 and so on

J_loss_hist = np.zeros([max_iter,1]) # empty array to store loss function values
J_loss_new  = 0 # initialize loss function value
J_loss      = cost_f(X,y,theta) # initial loss
err         = np.zeros([thetas,g]) # initialize errors

# Gradient Descent Algorithm

while J_loss - J_loss_new > Er:

 	J_loss_hist [no_iter,0] = cost_f(X,y,theta)

 	J_loss = cost_f(X,y,theta)

 	predictions = X.dot(theta).flatten()

 	for i in range(0,thetas):
 		err[i,:] = (predictions - y) * X[:,i]
 		theta[i,0] = theta[i,0] - alpha * (1.0/m) * err[i,:].sum()
 		theta_vals[no_iter,i] = theta[i,0]

 	J_loss_new = cost_f(X,y,theta)

 	no_iter = no_iter + 1
 		
# Compare gradient descent results with OLS

beta2 = (np.linalg.inv(X.T.dot(X))).dot(X.T).dot(y.T)

# Plot loss function and parameters

plt.plot(J_loss_hist[0:no_iter])
plt.title('Loss Function')
plt.show()

plt.plot(theta_vals[0:no_iter])
plt.title('Parameters')
plt.show()

print 'Algorithm Converged in', no_iter, 'Iterations!'

print 'Values from Gradient Descent'
print theta
print '================='
print 'Values from OLS'
print beta2