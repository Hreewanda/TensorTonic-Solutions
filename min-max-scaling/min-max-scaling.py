def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    # Write code here 
    import numpy as np 
    data = np.asarray(data, dtype = float)
    col_min = np.min(data, axis = 0)
    col_max = np.max(data, axis = 0)
    range_ = col_max - col_min 
    range_[range_ == 0] = 1.0 
    scaled = (data - col_min)/range_ 
    return scaled.tolist()