# -*- coding: utf-8 -*-
'''
practice on real data
'''
from sklearn.datasets import load_boston
import torch
import matplotlib.pyplot as plt

def linear_regression_loss_grad(theta, X, y):
    # theta, X and y have the same shape as used previously
    grad = X.t()@X@theta - X.t()@y
    return grad

X, y = tuple(torch.Tensor(z) for z in load_boston(True)) #convert to pytorch Tensors
X = X[:, [2,5]] # We're just going to use features 2 and 5, rather than using all of of them
X = torch.cat((X, torch.ones((X.shape[0], 1))), 1) # append a column of 1's to the X's
y = y.reshape(-1, 1) # reshape y into a column vector
print('X:', X.shape)
print('y:', y.shape)

# We're also going to break the data into a training set for computing the regression parameters
# and a test set to evaluate the predictive ability of those parameters
perm = torch.randperm(y.shape[0])
X_train = X[perm[0:253], :]
y_train = y[perm[0:253]]
X_test = X[perm[253:], :]
y_test = y[perm[253:]]

'''
用伪逆矩阵求参
'''
X_pinv = torch.pinverse(X_train)
theta = X_pinv @ y_train

assert(theta.shape == (3,1))

#print("Theta: ", theta.t())
print("MSE of test data on pinverse: ", torch.nn.functional.mse_loss(X_test @ theta, y_test))

'''
梯度下降求参
'''
alpha = 0.00001
theta_gd = torch.rand((X_train.shape[1], 1))
for e in range(0, 10000):
    gr = linear_regression_loss_grad(theta_gd, X_train, y_train)
    theta_gd -= alpha * gr

#print("Gradient Descent Theta: ", theta_gd.t())
print("MSE of test data on gradient descent: ", torch.nn.functional.mse_loss(X_test @ theta_gd, y_test))

perm = torch.argsort(y_test, dim=0)
plt.plot(y_test[perm[:,0]].numpy(), '.', label='True Prices')
plt.plot((X_test[perm[:,0]] @ theta).numpy(), '.', label='Predicted (pinv)')
plt.plot((X_test[perm[:,0]] @ theta_gd).numpy(), '.', label='Predicted (G.D.)')
plt.xlabel('House Number')
plt.ylabel('House Price ($,000s)')
plt.legend()
plt.show()

