# -*- coding: utf-8 -*-
import torch
import math

# we wouldn't normally do this, but for this lab we want to work in double precision
# as we'll need the numerical accuracy later on for doing checks on our gradients:
torch.set_default_dtype(torch.float64) 

def p_function(Theta, X, y, k, i):
    p = 0.
    for j in range(k):
        p += (math.exp(Theta[:, k].t()@X[:, i])) / math.exp(Theta[:, j].t()@X[:, i])
    
    return p

def softmax_regression_loss_grad(Theta, X, y):
    grad = 0.
    for i in range(X.shape[0]):
        
def softmax_regression_loss(Theta, X, y):
    