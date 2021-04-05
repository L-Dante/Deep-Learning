# -*- coding: utf-8 -*-
import torch

pred = torch.tensor([[-1.0, 2],[0.5, 1]])
label = torch.tensor([[1, 0.0],[2,5]])
mask = torch.tensor([[1,0], [1,1]])
l = torch.nn.functional.binary_cross_entropy_with_logits(pred, target=label, reduction="mean", weight=mask)
print(l)

