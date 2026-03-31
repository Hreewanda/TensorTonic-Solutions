import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points = np.asarray(points, dtype=float)
    assignments = np.asarray(assignments)
    
    n_features = points.shape[1]
    
    # Initialize sums and counts
    sums = np.zeros((k, n_features))
    counts = np.zeros(k)
    
    # Accumulate sums
    for i in range(len(points)):
        cluster = assignments[i]
        sums[cluster] += points[i]
        counts[cluster] += 1
    
    # Compute means
    centroids = []
    for i in range(k):
        if counts[i] == 0:
            centroids.append([0.0] * n_features)  # empty cluster
        else:
            centroids.append((sums[i] / counts[i]).tolist())
    
    return centroids