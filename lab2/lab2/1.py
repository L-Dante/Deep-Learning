# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:14:42 2021

@author: 12605
"""
from typing import Tuple
import torch

def sgd_factorise(A: torch.Tensor, r: int, num_epochs=1000, lr=0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    m = A.shape[0]
    n = A.shape[1]
    U = torch.rand(m, r, requires_grad=True)
    V = torch.rand(n, r, requires_grad=True)
    
    for epoch in range(num_epochs):
        e = torch.nn.functional.mse_loss(U@V.T, A, reduction='sum')
        e.backward()
        U = U - lr*U.grad
        U.retain_grad()
        V = V - lr*V.grad
        V.retain_grad()
        
    return U, V, e

'''
A = torch.Tensor([[0.3374, 0.6005, 0.1735],
                  [3.3359, 0.0492, 1.8374],
                  [2.9407, 0.5301, 2.2620]])
U, V, loss = sgd_factorise(A, 2)

print("loss: ", loss)
'''