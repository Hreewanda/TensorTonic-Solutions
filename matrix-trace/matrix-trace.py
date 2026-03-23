import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A = np.asarray(A, dtype=float)
    
    # Check square matrix
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    
    trace = 0.0
    for i in range(A.shape[0]):
        trace += A[i, i]
    
    return trace