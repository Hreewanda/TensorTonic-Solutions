import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred) 

    eps = 1e-15 
    y_pred = np.clip(y_pred, eps, 1 - eps)
    correct_probs = y_pred[np.arange(len(y_true)), y_true]
    loss = -np.mean(np.log(correct_probs)) 
    return loss