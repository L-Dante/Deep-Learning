# -*- coding: utf-8 -*-
import torch
import pandas as pd
import torch.optim as optim




df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases' + '/iris/iris.data', header = None)
df = df.sample(frac=1, random_state=0) # shuffle

df = df[df[4].isin(['Iris-virginica', 'Iris-versicolor'])] # filter

# add label indices column
mapping = {k: v for v, k in enumerate(df[4].unique())}
df[5] = (2 * df[4].map(mapping)) - 1 #labels in {-1, 1})
         
# normalise data
alldata = torch.tensor(df.iloc[:, [0,1,2,3]].values, dtype=torch.float)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)

# create datasets
targets_tr = torch.tensor(df.iloc[:75, 5].values, dtype=torch.long)
targets_va = torch.tensor(df.iloc[75:, 5].values, dtype=torch.long)
data_tr = alldata[:75]
data_va = alldata[75:]

from torch.utils import data
import numpy as np
dataset = data.TensorDataset(data_tr,targets_tr) # create your datset
dataloader = data.DataLoader(dataset, batch_size=25, shuffle=True) # create your dataloader


def svm(x, w, b):
    h = (w*x).sum(1) + b
    return h

def hinge_loss(y_pred, y_true):
    return torch.mean(torch.max(torch.zeros(len(y_pred)), 1-y_true*y_pred))

# SGD
w = torch.randn(1, 4, requires_grad=True)
b = torch.randn(1, requires_grad=True)
w1, b1 = w, b
opt1 = optim.SGD([w1,b1], lr=0.01, weight_decay=0.0001)
for epoch in range(100):
    for batch in dataloader:
        opt1.zero_grad()
        output1 = hinge_loss(svm(batch[0],w1,b1),batch[1])
        output1.backward()
        opt1.step()
target_pre1 = svm(data_va,w1,b1).detach().numpy()
tp1 = targets_va[np.where((target_pre1*targets_va.numpy()>1) & (targets_va.numpy()==1))]
tf1 = targets_va[np.where((target_pre1*targets_va.numpy()>1) & (targets_va.numpy()==-1))]
print("SGD validation accuracy:", (len(tf1)+len(tp1))/len(target_pre1))

# Adam
w2, b2 = w, b
opt2 = optim.Adam([w2,b2], lr=0.01, weight_decay=0.0001)
for epoch in range(100):
    for batch in dataloader:
        opt2.zero_grad()
        output2 = hinge_loss(svm(batch[0],w2,b2),batch[1])
        output2.backward()
        opt2.step()
target_pre2 = svm(data_va,w2,b2).detach().numpy()
tp2 = targets_va[np.where((target_pre2*targets_va.numpy()>1) & (targets_va.numpy()==1))]
tf2 = targets_va[np.where((target_pre2*targets_va.numpy()>1) & (targets_va.numpy()==-1))]
print("Adam validation accuracy:", (len(tf2)+len(tp2))/len(target_pre2))

