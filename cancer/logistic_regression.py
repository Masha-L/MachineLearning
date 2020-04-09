import numpy as np

def logistic(z):
	"""
	The logistic function
	Input:
	   z   numpy array (any shape)
	Output:
	   p   numpy array with same shape as z, where p = logistic(z) entrywise
	"""

	ones_z = np.ones(z.shape)
	exp = np.full(z.shape, np.e)
	denum = ones_z.copy() + np.power(exp, -z)
	p = np.divide(ones_z, denum)
	return p

def nll_cost_function(X, y, theta):
	"""
	Compute the negative log liklihood (nll) cost function for a particular data 
	set and hypothesis (weight vector)
	
	Inputs:
		X      data matrix (2d numpy array with shape m x n)
		y      label vector (1d numpy array -- length m)
		theta  parameter vector (1d numpy array -- length n)
	Output:
		cost   the value of the cost function (scalar)
	"""
	cost = 0
	h = logistic(np.dot(X,theta))
	cost = -np.dot(y, np.log(h)) - np.dot((1-y), np.log(1 - h))
	return cost


def gradient_descent( X, y, theta, alpha, iters ):
	"""
	Fit a logistic regression model by gradient descent.
	Inputs:
		X          data matrix (2d numpy array with shape m x n)
		y          label vector (1d numpy array -- length m)
		theta      initial parameter vector (1d numpy array -- length n)
		alpha      step size (scalar)
		iters      number of iterations (integer)
	Return (tuple):
		theta      learned parameter vector (1d numpy array -- length n)
		J_history  cost function in iteration (1d numpy array -- length iters)
	"""
	J_history = np.zeros(iters)

	for i in range(iters):
		d_J = 2*(np.dot(X.T, np.subtract(logistic(np.dot(X,theta)),y)))
		theta = theta - (alpha*d_J)
		J_history[i] = nll_cost_function(X, y, theta)
		
	return theta, J_history


