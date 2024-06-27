import numpy as np 

def iterative_A(A, max_iterations=3):
    '''
    calculate the largest eigenvalue of A
    '''
    x = A.sum(axis=1)
    #k = 3
    for _ in range(max_iterations):
        temp = np.dot(A, x)
        y = temp / np.linalg.norm(temp, 2)
        temp = np.dot(A, y)
        x = temp / np.linalg.norm(temp, 2)
    return np.dot(np.dot(x.T, A), y)