def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here 
    import numpy as np 
    x1 = np.array(x1)
    x2 = np.array(x2)
    dot = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    cos_sim = dot/(norm1*norm2)

    if label == 1:
        return 1 - cos_sim 
    else: 
        return max(0, cos_sim - margin)