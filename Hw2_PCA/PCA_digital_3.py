# -*- coding: utf-8 -*-
# perform PCA over all digit '3' with 2 components in python
import numpy as np
import matplotlib.pyplot as plt 

#PCA
def pca():
    Mat = []
    file_object = open("optdigits-orig.tra")
    Xi = [] 
    while 1:
        lines = file_object.readlines(63850)   
        if not lines:
            break
        for line in lines:
            line=line.strip('\n')
            if len(line) < 5 :
                number = int(line)
                if  number == 3:
                    Mat.append(Xi)  
                Xi = []
            else:
                for str in line:
                    Xi.append(int(str))
    Matrix = np.matrix(Mat).T
    MatrixCenter = Matrix - Matrix.mean(1)
    U, s, V = np.linalg.svd(MatrixCenter, full_matrices=True) 
    Matrix_from_U = np.matrix(np.array([(U[:,0]), (U[:,1])])).T
    Matrix_lenda = Matrix_from_U.getI()* MatrixCenter
    Matrix_1 = Matrix_lenda.T.tolist()
    X = [ele[0] for ele in Matrix_1] 
    Y = [ele[1] for ele in Matrix_1] 
    return X,Y
 
  
def show():
    X,Y = pca()
    X1 = [-9,9]
    Y1 = [0,0]
    X2 = [0,0]
    Y2 = [-9,9]
    plt.plot(X,Y,'ro',label = 'Digital_3')
    plt.plot(X1,Y1,'k-')
    plt.plot(X2,Y2,'k-')
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("PCA")
    plt.xlim(-7,8.5)
    plt.ylim(-8,8)
    plt.legend()
    plt.savefig('PCA_digital_3.png')
    plt.show()

show()