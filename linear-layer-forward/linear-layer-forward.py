def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    import numpy as np 
    X = np.asarray(X)
    W = np.asarray(W)
    b = np.asarray(b) 

    Y = X @ W 
    Y = Y + b 
    return Y.tolist()

 
