import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return x*(x>0)

def test(x):
    return(x+1)

# X=np.array([[1,2,1],[1,5,1]])
# W1=np.array([[-1,0,1],[0,-1,0],[1,0,-1]])
# W2=np.array([[-1,0,1],[0,-1,0],[1,0,1],[1,-1,1]])
# f = np.vectorize(sigmoid)
# layer1=f(X@W1)
# print('f(X@W1)',layer1)
# inner2=np.hstack((np.ones((len(layer1),1)),layer1))
# layer2=f(inner2@W2)
# print('f(inner2)',layer2)
# inner3=np.hstack((np.ones((len(layer2),1)),layer2))
# layer3=f(inner3@W2)
# print('layer3', layer3)


# X=np.array([[1,1,3],[1,2,2.5]])
# W1=np.array([[-1,0,1],[0,-1,0],[1,0,1]])
# f = np.vectorize(ReLU)
# print(f(f(X@W1)@W1))