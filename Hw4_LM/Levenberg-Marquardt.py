# -*- coding: utf-8 -*-
#Implement the Levenberg-Marquardt method in python 
import numpy as np

#calculate the g  
def count_g(X):
    n = X.size
    g = np.zeros(n)
    g.shape = (n, 1)
    for i in range(X.size):
        if i == 0:
            g[i] =  np.cos(X[0]) + X[0] / 5.
        elif i == 1:
            g[i] =  np.cos(X[1]) + X[1] / 5.
        else:
            g[i] =  0.0
    return g
    
#calculate the G
def count_G(X):
    n = X.size
    G = np.mat(np.arange(0.0, 1.0 * n * n, 1))
    G.shape = (n, n)
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                G[i, j] = -np.sin(X[0]) + 1. / 5.
            elif i == 0 and j == 1:
                G[i, j] = 0.
            elif i == 1 and j == 0:
                G[i, j] = 0.
            elif i == 1 and j == 1:
                G[i, j] = - np.sin(X[1]) + 1. / 5.
            else:
                G[i, j] = 0.0
    return G
    
def func_1(X):
    return np.sin(X[0]) + np.sin(X[1]) + (X[0] * X[0] + X[1] * X[1])/10. 
    
def def_pos(m):
    w, v = np.linalg.eig(m)
    t = w.min()
    if t >= 0: return 1
    else: return 0

def LM():
    #initialize the start point   
    x_origin = 2.0
    y_origin = -7.0
    u = 1.0
    x = np.array([x_origin, y_origin])
    x = x.reshape(x.size, 1)
    f = func_1(x)
    g = count_g(x)
    G = count_G(x)
    e = 0.00000001
    while np.vdot(g, g) >= e * e:
        G_Gen = G.copy()
        while def_pos(G_Gen + np.identity(x.size) * u) == 0:
            u *= 4
        G_Gen += np.identity(x.size) * u
        A = np.mat(G_Gen)
        B = np.mat(-g)
        s = np.linalg.solve(A, B)
        func_2 = func_1(x + s)
        df = func_2 - f
        dq = np.dot(g.T, s) + 0.5 * np.dot(np.dot(s.T, G), s)
        rk = dq / df
        if rk < 0.25:
            u *= 4
        elif rk > 0.75:
            u *= 0.5
        if rk > 0.0:
            xi2 = (x[0] + s[0])
            yi2 = (x[1] + s[1])
            x2 = np.array([xi2, yi2])
            x2 = x2.reshape(x2.size, 1)
            x = x2
            f = func_2
        g = count_g(x)
        G = count_G(x)
    print x
    print 'gk = ' + str(g)
    print 'Gk = ' + str(G)

LM()
        


