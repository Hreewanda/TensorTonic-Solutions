import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X, dtype = float)
    if X.ndim != 2 or X.shape[0] < 2:
        return None 
    N = X.shape[0]

    mean = np.mean(X, axis = 0)
    X_centered = X - mean 
    cov = (X_centered.T @ X_centered)/(N - 1)

    return cov
    
