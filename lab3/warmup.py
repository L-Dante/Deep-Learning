# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import torch
import torch.optim as optim

def himm(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


xmin, xmax, xstep = -5, 5, .2
ymin, ymax, ystep = -5, 5, .2
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = himm(torch.tensor([x, y])).numpy()

fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d', elev=50, azim=-50)
ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, 
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

plt.show()

xmin, xmax, xstep = -5, 5, .2
ymin, ymax, ystep = -5, 5, .2
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = himm(torch.tensor([x, y])).numpy()


fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.gray)

p = torch.tensor([[0.0],[0.0]], requires_grad=True)
p_mom = torch.tensor([[0.0],[0.0]], requires_grad=True)
opt = optim.SGD([p], lr=0.01)
opt_mom = optim.SGD([p_mom], lr=0.01, momentum=0.9)

path = np.empty((2,0))
path = np.append(path, p.data.numpy(), axis=1)
path_mom = np.empty((2,0))
path_mom = np.append(path_mom, p_mom.data.numpy(), axis=1)

for i in range(50):
    opt.zero_grad()
    output = himm(p)
    output.backward()
    opt.step()
    path = np.append(path, p.data.numpy(), axis=1)
    
# set p to zero
#p = torch.tensor([[0.0],[0.0]], requires_grad=True)
    
for i in range(50):
    opt_mom.zero_grad()
    output_mom = himm(p_mom)
    output_mom.backward()
    opt_mom.step()
    path_mom = np.append(path_mom, p_mom.data.numpy(), axis=1)
    

ax.plot(path[0], path[1], color='green', label='SGD', linewidth=2)
ax.plot(path_mom[0], path_mom[1], color='yellow', label='SGD_mom', linewidth=2)

ax.legend()
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))