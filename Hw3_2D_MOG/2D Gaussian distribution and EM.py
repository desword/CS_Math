# -*- coding: utf-8 -*-
# 2D Gaussian distribution and EM in python by

from matplotlib import pyplot as plt
from numpy.linalg import det
from pylab import *
import numpy as np
import numpy.matlib as matlib
import random

#generate the gaussian distribution
def gauss():
    mean = [0,0]
    cov = [[1,0],[0,100]] #Diagonal covariance
    x,y = np.random.multivariate_normal(mean,cov,1000).T
    return x, y

#calcualte the distance from the original point.
def distance(X, Y):
    n = len(X)
    m = len(Y)
    xx = matlib.sum(X*X, axis=1)
    yy = matlib.sum(Y*Y, axis=1)
    xy = matlib.dot(X, Y.T)
    return tile(xx, (m, 1)).T+tile(yy, (n, 1)) - 2*xy

def params(centers,k):
    Miu = centers
    Pi = zeros([1,k], dtype=float)
    Sigma = zeros([len(X[0]), len(X[0]), k], dtype=float)
    dist = distance(X, centers)
    labels = dist.argmin(axis=1) #the min distance from the original
    for j in range(k):
        idx_j = (labels == j).nonzero()
        Miu[j] = X[idx_j].mean(axis=0)
        Pi[0, j] = 1.0 * len(X[idx_j]) / N
        Sigma[:, :, j] = cov(mat(X[idx_j]).T)
    return Miu, Pi, Sigma

def prob(k,Miu,Sigma):
    Px = zeros([N, k], dtype=float)
    for i in range(k):
        Xshift = mat(X - Miu[i, :])
        inv_pSigma = mat(Sigma[:, :, i]).I
        coef = pow((2*pi), (len(X[0])/2)) * sqrt(det(mat(Sigma[:, :, i])))
        for j in range(N):
            tmp = (Xshift[j, :] * inv_pSigma * Xshift[j, :].T)
            Px[j, i] = 1.0 / coef * exp(-0.5*tmp)
    return Px

def observer(iter, labels):
    print "iter %d." % iter
    colors = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    plt.plot(hold=False)
    plt.hold(True)
    labels = array(labels).ravel()
    data_colors=[colors[lbl] for lbl in labels]
    plt.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
    plt.savefig("EM iteration = " + str(iter) + " .png")

#plot the picture. 
def show(X, labels,iter):
    colors = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    plt.plot(hold=False)
    plt.hold(True)
    labels = array(labels).ravel()
    data_colors=[colors[lbl] for lbl in labels]
    plt.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
    plt.savefig("EM iteration = " + str(iter) + " .png")
    
#implemnet MoG
def MoG(X, k, observer=None, threshold=1e-15, maxiter=300):
    N = len(X)
    labels = zeros(N, dtype=int)
    centers = array(random.sample(X, k))
    iter = 0
    Miu, Pi, Sigma = params(centers,k)
    Lprev = float('-10000')
    pre_esp = 100000
    while iter < 80: #the max iterative loop is 80 
        Px = prob(k,Miu,Sigma)
        pGamma =mat(array(Px) * array(Pi))
        pGamma = pGamma / pGamma.sum(axis=1) #updat the pGamma ([N, k])
        Nk = pGamma.sum(axis=0)
        Miu = diagflat(1/Nk) * pGamma.T * mat(X)
        Pi = Nk / N
        Sigma = zeros([len(X[0]), len(X[0]), k], dtype=float)
        for j in range(k):
            Xshift = mat(X) - Miu[j, :]
            for i in range(N):
                SigmaK = Xshift[i, :].T * Xshift[i, :]
                SigmaK = SigmaK * pGamma[i, j] / Nk[0, j]
                Sigma[:, :, j] = Sigma[:, :, j] + SigmaK
        labels = pGamma.argmax(axis=1)
        iter = iter + 1
        L = sum(log(mat(Px) * mat(Pi).T)) #converage the condition
        esp = L-Lprev
        if esp < threshold:
            break
        if esp > pre_esp:
            break
        pre_esp=esp
        Lprev = L
        print "iteration " + str(iter) + " : " + str(esp)
    show(X, labels,iter)

samples = gauss()
N = len(samples[0])
X = zeros((N, 2))
for i in range(N):
    X[i, 0] = samples[0][i]
    X[i, 1] = samples[1][i]
MoG(X, 3, observer=observer)