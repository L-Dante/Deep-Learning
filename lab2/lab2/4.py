# -*- coding: utf-8 -*-
import pandas as pd
import torch

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases' + '/iris/iris.data', header = None)
df = df.sample(frac=1) #shuffle

# add label indices column
mapping = {k: v for v, k in enumerate(df[4].unique())}
df[5] = df[4].map(mapping)

# normalise data
alldata = torch.tensor(df.iloc[:, [0,1,2,3]].values, dtype=torch.float)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)

# create datasets
targets_tr = torch.tensor(df.iloc[:100, 5].values, dtype=torch.long)
targets_va = torch.tensor(df.iloc[:100, 5].values, dtype=torch.long)
data_tr = alldata[:100]
data_va = alldata[100:]

W1 = torch.randn(4,12,requires_grad = True)
W2 = torch.randn(12,3,requires_grad = True)
b1 = torch.zeros(1,requires_grad = True)
b2 = torch.zeros(1,requires_grad = True)
#logits = torch.relu(alldata@W1.float() + b1)@W2 + b2
for i in range(100):
    logits = torch.relu(data_tr@W1.float() + b1) @ W2 + b2
    loss = torch.nn.functional.cross_entropy(logits, targets_tr)
    loss.backward()
    W1.data -= 0.01 * W1.grad
    W2.data -= 0.01 * W2.grad
    b1.data -= 0.01 * b1.grad
    b2.data -= 0.01 * b2.grad
    W1.grad.zero_()
    W2.grad.zero_()
    b1.grad.zero_()
    b2.grad.zero_()

print(loss)