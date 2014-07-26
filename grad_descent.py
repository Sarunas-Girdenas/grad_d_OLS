"""This is the simple implementation of a batch gradient descent algorithm.
Partially inspired by https://gist.github.com/marcelcaraciolo/1321575
Written by Sarunas Girdenas, sg325@exeter.ac.uk, July, 2014"""

import numpy as np
import matplotlib.pyplot as plt

# Import some data

Data = np.genfromtxt('some_data_file.txt',usecols=(0,1),delimiter=',',dtype=None)

# Name the first and second variables

Stock_Pr = Data[0:len(Data),0]
Index_Pr = Data[0:len(Data),1]

# clear redundand Data

del Data

# Let's do gradient descent now

g = Stock_Pr.size # size of the sample

theta = np.zeros([2,1]) # container for theta

theta[0,0] = 5
theta[1,0] = 3 # initial guesses for theta

alpha = 0.01 # learning rate (step of algorithm)

# set the hypothesis, in our case it is:
# y = theta_0 + theta_1*X
# Index_pr = alpha + beta*Stock_Pr

X = np.array([Stock_Pr])
y = np.array([Index_Pr])

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
Er       = 0.00000001 # tolerance for error

theta_0 = np.zeros([max_iter,1]) # empty array to store theta_0
theta_1 = np.zeros([max_iter,1]) # empty array to store theta_1	

J_loss_hist = np.zeros([max_iter,1]) # empty array to store loss function values
J_loss_new  = 0 # initialize loss function value
J_loss      = cost_f(X,y,theta) # initial loss

# Gradient Descent Algorithm

while J_loss - J_loss_new > Er:

	no_iter = no_iter + 1

	J_loss_hist [no_iter,0] = cost_f(X,y,theta)

	J_loss = cost_f(X,y,theta)

	predictions = X.dot(theta).flatten()

	err_x1 = (predictions - y) * X[:,0]
	err_x2 = (predictions - y) * X[:,1]

	theta[0][0] = theta[0][0] - alpha * (1.0/m) * err_x1.sum()
	theta[1][0] = theta[1][0] - alpha * (1.0/m) * err_x2.sum()

	J_loss_new = cost_f(X,y,theta)

	theta_0[no_iter][0] = theta[0][0]
	theta_1[no_iter][0] = theta[1][0]


# Compare gradient descent results with OLS

X2 = np.concatenate((np.ones([g,1]),np.array([Stock_Pr]).T),axis=1)

beta2 = (np.linalg.inv(X2.T.dot(X2))).dot(X2.T).dot(y.T)

print 'Algorithm Converged in', no_iter, 'Iterations!'

print 'Values from Gradient Descent'
print theta
print '================'
print 'Values from OLS'
print beta2

# Do the scatter plot for the raw data

plt.scatter(Stock_Pr,Index_Pr)
plt.xlabel('Stock Price')
plt.ylabel('Index Price')
plt.title('Scatter Plot of Raw Data')
plt.show()

# Construct the array of predicted values
X3 = np.zeros([g,2])
X3[:,0] = np.ones([g])
X3[:,1] = np.asarray(range(1,Stock_Pr.size+1))
Y_pred = X3.dot(theta)

# Plot the fitted line on the data

plt.scatter(Stock_Pr,Index_Pr)
plt.plot(Y_pred,'r')
plt.xlabel('Stock Price')
plt.ylabel('Index Price')
plt.title('Scatter Plot of Variables and Fitted Line')
plt.show()

