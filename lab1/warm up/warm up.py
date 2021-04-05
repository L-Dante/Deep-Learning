# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:59:49 2021

@author: 12605
"""

import torch
import matplotlib.pyplot as plt

'''
用伪逆矩阵求参
'''
# Generate some data points on a straight line perturbed with Gaussian noise
N = 1000 # number of points
theta_true = torch.Tensor([[1.5], [2.0]]) # true parameters of the line 真实参数

X = torch.rand(N, 2)
X[:, 1] = 1.0
y = X @ theta_true + 0.1 * torch.randn(N, 1) # Note that just like in numpy '@' represents matrix multiplication and A@B is equivalent to torch.mm(A, B)

plt.scatter(X[:,0].numpy(), y.numpy())
plt.show()

# direct solution using moore-penrose pseudo inverse
X_inv = torch.pinverse(X)
theta_pinv = torch.mm(X_inv, y) #伪逆矩阵
#print(theta_pinv)

'''
用svd求参
'''
u, s, v = torch.svd(X) #s是对角矩阵的对角元素，使用时需要扩展成矩阵
'''
求s的伪逆矩阵
将对角线上的元素取倒数
再将整个矩阵转置一次
'''
for i in range(len(s)):
    s[i] = 1/s[i]

s = torch.diag(s).t()
X_inv_svd = v@s@u.t()

theta_pinv_svd = torch.mm(X_inv_svd, y)
#print(theta_pinv_svd)
assert(torch.all(torch.lt(torch.abs(torch.add(theta_pinv, -theta_pinv_svd)), 1e-6)))

'''
梯度下降求参
'''
def linear_regression_loss_grad(theta, X, y):
    # theta, X and y have the same shape as used previously
    grad = X.t()@X@theta - X.t()@y
    return grad

assert(linear_regression_loss_grad(torch.zeros(2,1), X, y).shape == (2,1))
alpha = 0.001
theta = torch.Tensor([[0], [0]])
for e in range(0, 200):
    gr = linear_regression_loss_grad(theta, X, y)
    theta -= alpha * gr

#print(theta)

