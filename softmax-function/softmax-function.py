import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.asarray(x, dtype=float)
    
    # 1D case
    if x.ndim == 1:
        x_shifted = x - np.max(x)   # stability
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    # 2D case (row-wise)
    elif x.ndim == 2:
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)