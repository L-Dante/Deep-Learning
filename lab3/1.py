# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import cm 
import torch
import torch.optim as optim
import math

def rastrigin(x):
    A = 1
    return (x[0]**2 - A * torch.cos(2 * math.pi * x[0])) + (x[1]**2 - A * torch.cos(2 * math.pi * x[1])) + 2*A

'''
xmin, xmax, xstep = -5.12, 5.12, .2
ymin, ymax, ystep = -5.12, 5.12, .2
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = rastrigin(torch.tensor([x, y])).numpy()
              
fig = plt.figure() 
ax = fig.gca(projection='3d') 
ax.plot_surface(x, y, z, rstride=1, cstride=1,cmap=cm.nipy_spectral, linewidth=0.08, antialiased=True)    
plt.show()
'''
A=1

xmin, xmax, xstep = -5.12, 5.12, .2
ymin, ymax, ystep = -5.12, 5.12, .2
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = rastrigin(torch.tensor([x, y])).numpy()

# first graph
fig = plt.figure() 
ax = fig.gca(projection='3d') 
ax.plot_surface(x, y, z, rstride=1, cstride=1,cmap=cm.nipy_spectral, linewidth=0.08, antialiased=True)    
plt.show()


# SGD
p = torch.tensor([[5.0],[5.0]], requires_grad=True)
opt = optim.SGD([p], lr=0.01)
path = np.empty((2,0))
path = np.append(path, p.data.numpy(), axis=1)

for i in range(100):
    opt.zero_grad()
    output = rastrigin(p)
    output.backward()
    opt.step()
    path = np.append(path, p.data.numpy(), axis=1)

# second graph
fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.gray)
ax.plot(path[0], path[1], color='green', label='SGD', linewidth=2)
ax.legend()
ax.set_title('SGD')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
loss = torch.nn.functional.mse_loss(p, torch.tensor([[0.0], [0.0]]), reduction='sum')
print("result of SGD")
print(p.data)



# SGD + momentum
p = torch.tensor([[5.0],[5.0]], requires_grad=True)
opt = optim.SGD([p], lr=0.01, momentum=0.9)
path = np.empty((2,0))
path = np.append(path, p.data.numpy(), axis=1)

for i in range(100):
    opt.zero_grad()
    output = rastrigin(p)
    output.backward()
    opt.step()
    path = np.append(path, p.data.numpy(), axis=1)

# second graph
fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.gray)
ax.plot(path[0], path[1], color='green', label='SGD_mom', linewidth=2)
ax.legend()
ax.set_title('SGD+momentum')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
print("result of SGD+momentum")
print(p.data)


# Adagrad
p = torch.tensor([[5.0],[5.0]], requires_grad=True)
opt = optim.Adagrad([p], lr=0.01)
path = np.empty((2,0))
path = np.append(path, p.data.numpy(), axis=1)

for i in range(100):
    opt.zero_grad()
    output = rastrigin(p)
    output.backward()
    opt.step()
    path = np.append(path, p.data.numpy(), axis=1)

# second graph
fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.gray)
ax.plot(path[0], path[1], color='green', label='Adagrad', linewidth=2)
ax.legend()
ax.set_title('Adagrad')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
print("result of Adagrad")
print(p.data)


# Adam
p = torch.tensor([[5.0],[5.0]], requires_grad=True)
opt = optim.SGD([p], lr=0.01)
path = np.empty((2,0))
path = np.append(path, p.data.numpy(), axis=1)

for i in range(100):
    opt.zero_grad()
    output = rastrigin(p)
    output.backward()
    opt.step()
    path = np.append(path, p.data.numpy(), axis=1)

# second graph
fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.gray)
ax.plot(path[0], path[1], color='green', label='Adam', linewidth=2)
ax.legend()
ax.set_title('Adam')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
print("result of Adam")
print(p.data)