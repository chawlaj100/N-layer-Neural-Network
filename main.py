import numpy as np
from model import *
X = np.array([[0,0],[1,1],[0,1],[1,0]]) # Input X
Y = np.array([[1],[1],[0],[0]])         # Ouput Y
weights = initialize_params([2,5,1])    # Initializing the parameters with 2 nodes for input layer, 5 nodes for hidden layer, and 1 node for output.
for i in range(50000):                  # You can change the number of iterations according to your problem
    A_f, caches = forward(X,weights)
    
    cost = compute_cost(A_f,Y)
    if(i%5000 == 0):
        print(cost)
    
    grads = backward(A_f, Y, caches)

    weights = update_parameters(weights, grads)


print("finished training..")
print(weights)
X = np.array([[0,1],[1,1]])
ans = predict(X, weights)
print(X,ans)
