import numpy as np

def r2_score(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute mean of true values
    y_mean = np.mean(y_true)
    
    # Total sum of squares
    sst = np.sum((y_true - y_mean) ** 2)
    
    # Residual sum of squares
    sse = np.sum((y_true - y_pred) ** 2)
    
    # Handle constant target case
    if sst == 0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    
    # Compute R²
    return 1 - (sse / sst)