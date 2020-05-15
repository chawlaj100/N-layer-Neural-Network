# Here all the functions for forming a complete Feed-Forward are defined and can be used in the main.py
import numpy as np
import utils
# weights = {}
grads = {}
def initialize_params(size):    # Used for initializing the parameters/weights of the network
    j=1
    weights = {}
    #np.random.seed(2)
    for i in range(len(size)-1):
        weights['W'+str(j)] = np.random.rand(size[i+1],size[i])*0.1
        weights['b'+str(j)] = np.random.rand(size[i+1],1)*0.1
        j+=1

    return weights

def forward_prop(X,W,b):    # Used for calculating the combination of the weights with the previous layer values
    #n = X.shape[0]
  
    A_next = np.dot(X,W.T) + b.T
    cache = X,W,b
    
    return A_next,cache

def forward_activate(X,W,b,act="sigmoid"):    # Used to apply the activations to the calculated values in the current layer
    if act=="relu":
        Z,forward_cache = forward_prop(X,W,b)
        A = utils.relu(Z)
        Z_cache = Z 
    elif act=="sigmoid":
        Z,forward_cache = forward_prop(X,W,b)
        A = utils.sigmoid(Z)
        Z_cache = Z
    cache = (A,Z_cache,forward_cache)
    
    return A,cache 

def forward(X,weights):    # Used to sum up the above 2 functions and returns the final layer output and completes one iteration of forward-prop
    caches = []
    num = len(weights)//2
    A_p = X
    for i in range(num-1):
        A_n,cache = forward_activate(A_p,weights['W'+str(i+1)],weights['b'+str(i+1)],"relu")
        A_p = A_n
        caches.append(cache)
    A_n,cache = forward_activate(A_p,weights['W'+str(num)],weights['b'+str(num)],"sigmoid")
    A_p = A_n
    caches.append(cache)
    return A_p,caches

def compute_cost(Z,Y):     # Used to compute the cost in between the given output and the output by the forward function. I have used MSE-loss.
    m = Z.shape[0]
    mediate = np.sum((Z-Y)**2)/m
    #cost = np.sum(mediate,axis=0)/m
    return mediate

def backward_prop(dZ,cache):    # Used for calculation of derivatives/gradients of the loss wrt. the weights and biases 
    A_p,W,b = cache
    #m = A_p.shape[0]
    dW = np.dot(dZ.T,A_p)
    db = np.sum(dZ,axis=0,keepdims=True)
    dA = np.dot(dZ,W)

    return dA,dW,db

def backward_activate(dA,cache,act="sigmoid"):     # Used to calculate the derivatives of the activations that are applied
    A,Z_cache,answers = cache
    dZ = np.zeros((dA.shape[0],dA.shape[1]))
    if act=="relu":
        dZ[A>0]=1
        dZ[A<=0]=0
    
    elif act=="sigmoid":
        dZ = np.multiply(utils.sigmoid(A),1 - utils.sigmoid(A))

    dZ = np.multiply(dA,dZ)
    dA_p,dW,db = backward_prop(dZ,answers)       
    
    return dA_p,dW,db

def backward(A_f,Y,caches):     # Used to combine the above 2 functions and return the gradients of the respective layer wrt. the weights and biases 
    num = len(caches)
    dA = 2*(A_f - Y)
    
    cache = caches[num-1]
    dA_p,dW,db = backward_activate(dA,cache,"sigmoid")
    dA, grads["dW"+str(num)], grads["db"+str(num)] = dA_p, dW, db

    for i in range(num-1,0,-1):
        cache = caches[i-1]
        dA_p,dW,db = backward_activate(dA,cache,"relu")
        dA, grads["dW"+str(i)], grads["db"+str(i)] = dA_p, dW, db

    return grads


def update_parameters(weights, grads, learning_rate=0.1):     # Used for updating the parameters in the directions where the loss function gets minimized according to the gradients calculated
    n = len(weights) // 2
    for i in range(1,n):
        weights["W"+str(i)] = weights["W"+str(i)] - learning_rate*grads["dW"+str(i)]
        #print(grads["db"+str(i)].shape)
        weights["b"+str(i)] = weights["b"+str(i)] - learning_rate*grads["db"+str(i)].T
        #print("after update")
        #print(weights["b"+str(i)])
    return weights


def predict(X,weights):     # Used for predicting the output after the updation of weights to check the results.
    n = len(weights)//2
    for i in range(0,n-1):
        Z = np.dot(X,weights["W"+str(i+1)].T) + weights["b"+str(i+1)].T
        A = utils.relu(Z)
        X=A
    Z = np.dot(X,weights["W"+str(n)].T) + weights["b"+str(n)].T
    A = utils.sigmoid(Z)
    return A