# -*- coding: utf-8 -*-
from typing import Tuple
import torch

def sgd_factorise(A: torch.Tensor, r: int, num_epochs=1000, lr=0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    m = A.shape[0]
    n = A.shape[1]
    U = torch.rand(m, r)
    V = torch.rand(n, r)
    for epoch in range(num_epochs):
        for r in range(m):
            for c in range(n):
                e = A[r, c] - U[r, :] @ V[c, :].T
                U[r, :] = U[r, :] + lr*e*V[c, :]
                V[c, :] = V[c, :] + lr*e*U[r, :]
    return U, V

A = torch.Tensor([[0.3374, 0.6005, 0.1735],
                  [3.3359, 0.0492, 1.8374],
                  [2.9407, 0.5301, 2.2620]])
U, V = sgd_factorise(A, 2)
loss = torch.nn.functional.mse_loss(U@V.T, A, reduction='sum')
print("loss: ", loss)