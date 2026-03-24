import numpy as np

def pearson_correlation(X):
    X = np.asarray(X, dtype=float)
    
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    
    N = X.shape[0]
    
    X_centered = X - np.mean(X, axis=0)
    
    cov = (X_centered.T @ X_centered) / (N - 1)
    
    std_devs = np.std(X, axis=0, ddof=1)
    
    denom = np.outer(std_devs, std_devs)
    
    corr = cov / denom
    
    # Handle zero variance
    corr[denom == 0] = np.nan
    
    # Only set diagonal where std != 0
    for i in range(len(std_devs)):
        if std_devs[i] != 0:
            corr[i, i] = 1.0
    
    return corr.tolist()