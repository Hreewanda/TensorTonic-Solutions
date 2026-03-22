import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    if rng is None:
        rng = np.random 
    mask = rng.random(x.shape) < (1-p) 
    scale = 1.0/(1-p) if p < 1 else 0.0 
    dropout_pattern = mask.astype(float) * scale 
    output = x * dropout_pattern 
    return output, dropout_pattern