# -*- coding: utf-8 -*-
import pandas as pd
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from typing import Tuple

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

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases' + '/iris/iris.data', header = None)
data = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values)
data = data - data.mean(dim=0)
U, V, loss = sgd_factorise(data, 2)
print("loss of gd_factorise_ad:")
print(loss)
U_svd, S_svd, V_svd = torch.svd(data)
S_svd[-1] = 0
S_svd[-2] = 0
loss_svd = torch.nn.functional.mse_loss(U_svd@torch.diag(S_svd)@V_svd.T, data, reduction='sum')
print("loss of svd:")
print(loss_svd)