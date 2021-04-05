# -*- coding: utf-8 -*-
import torch

A = torch.Tensor([[0.3374, 0.6005, 0.1735],
                  [3.3359, 0.0492, 1.8374],
                  [2.9407, 0.5301, 2.2620]])
U, S, V = torch.svd(A)
S[-1] = 0
loss = torch.nn.functional.mse_loss(U@torch.diag(S)@V.T, A)
print(loss)
