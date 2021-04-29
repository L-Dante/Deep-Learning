# -*- coding: utf-8 -*-
import torch

a = torch.tensor([2.], requires_grad=True)
b = torch.tensor([5.], requires_grad=True)

Q1 = torch.log(a)
Q2 = a * b
Q3 = torch.sin(b)
Q = Q1 + Q2 - Q3
print("Q: ", Q)

#external_grad = torch.tensor([1.])
Q.backward()


