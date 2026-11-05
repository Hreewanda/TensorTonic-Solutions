import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    a = np.asarray(a, dtype = float)
    b = np.asarray(b, dtype = float)
    y = np.asarray(y, dtype = float).reshape(-1)
    if a.ndim == 1:
        a = a.reshape(1,-1) 
    if b.ndim == 1:
        b = b.reshape(1, -1)
    d = np.linalg.norm(a-b, axis = 1)
    loss = y *(d**2) + (1-y)*np.maximum(0, margin -d)**2 
    if reduction == "mean":
        return float(np.mean(loss)) 
    elif reduction == "sum":
        return float(np.sum(loss))
    pass