# All the activation functions are defined here and can be used in model.py

import numpy as np
def relu(a):
    return np.maximum(a,0)

def sigmoid(a):
    return 1/1+np.exp(-a)

def softmax(a):
    probs = np.sum(a,axis=1)
    ans = probs/np.sum(probs)
    return ans