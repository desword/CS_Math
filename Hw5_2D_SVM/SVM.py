# -*- coding: utf-8 -*-
# Implement (simplified) SVM method in python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def func(w, *args):
    X,Y,c = args
    yp = np.dot(X,w)
    idx = np.where(yp * Y < 1)[0]
    e = yp[idx] - Y[idx]
    cost = np.dot(e, e) + c * np.dot(w,w)
    grad = 2 * (np.dot(X[idx].T, e) + c * w)
    return cost, grad
    
x = np.loadtxt('x.txt')
y = np.loadtxt('y.txt')

X = np.append(np.ones(x.shape[0], 1), x, 1)
Y = y
c = 0.001

#optimization algorithm L-BFGS-B
RET = opt.fmin_l_bfgs_b(func, x0 = np.random.rand(X.shape[1], args = (X,Y,c), approx_grad = False))

w = RET[0]
margin = 2/np.sqrt(np.dot(w[1:3],w[1:3]))
plot_x = np.append(np.min(x,0)[0] - 0.2, np.max(x,0)[0] + 0.2)
plot_y = -(plot_x * w[1] + w[0])/w[2]

plt.figure()
pos = (Y==1)
neg = (Y==-1)
plt.plot(x[pos][:,0], x[pos][:,1], "r+", label = "Positive Samples")
plt.plot(x[neg][:,0], x[neg][:,1], "bo", label = "Negative Samples")
plt.plot(plot_x, plot_y, "r-", label = "Decision boundary")
plt.plot(plot_x, plot_y + margin/2, "g-", label = "")
plt.plot(plot_x, plot_y - margin/2, "g-", label = "")
plt.xlabel("x1")
plt.xlabel("x2")
plt.title("SVM")
plt.legend()
plt.show()
