import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.asarray(x)
    x = np.clip(x, -500,500)
    sigmoid = 1/(1+ np.exp(-x))
    return x *sigmoid