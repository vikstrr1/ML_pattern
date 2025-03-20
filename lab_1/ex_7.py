import numpy as np


def ex7a_v1(m,n):
    M = np.zeros((m,n), dtype= np.float64)
    for i in range(m):
        for j in range(n):
            M[i,j] = i*j
    return M

def ex7b(M):
    sum = M.sum(0)
    return M/sum.reshape((1,M.shape[1]))

def ex7c(M):
    sum = M.sum(1)
    return M/sum.reshape((M.shape[0],1))

def ex7d(M):
    M_new = M.copy()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j] < 0:
                M_new[i,j] = 0
    return M_new
    



ex7a_v1(3,4)
M = M = np.array([[1.0, 2.0, 6.0, 4.0],
                  [3.0, 4.0, 3.0, 7.0],
                  [1.0, 4.0, 6.0, 9.0]])
    
 
print(ex7b(M))
print()
print(ex7c(M))
print()
M = np.array([[-1.0, 2.0, 3.0],
                     [2.0, -3.0, -4.0]])
print(ex7d(M))
print()
print(ex7d(M))