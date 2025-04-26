import numpy as np

def unbroadcast(grad, target_shape):
    # First, check if the gradient and target shape are broadcastable
    if grad.shape == target_shape:
        return grad

    # Reduce extra dimensions (when grad has more dimensions than the target shape)
    while len(grad.shape) > len(target_shape):
        grad = np.sum(grad, axis=0)
        

    # Then, sum along axes where target_shape has 1 (broadcasted dims)
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] != 1:  # We only sum over dimensions that are 1 in the target shape
            grad = np.sum(grad, axis=i, keepdims=True)

    return grad
