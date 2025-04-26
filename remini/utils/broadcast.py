import numpy as np

def unbroadcast(grad, target_shape):
    # First, check if the gradient and target shape are broadcastable
    try:
        np.broadcast(grad, np.empty(target_shape))  # Check if the grad and target_shape can be broadcasted
    except ValueError:
        raise ValueError(f"Cannot broadcast grad of shape {grad.shape} to target shape {target_shape}")

    # Now reduce extra dimensions
    while len(grad.shape) > len(target_shape):
        grad = np.sum(grad, axis=0, keepdims=True)

    # Then sum along axes where target_shape has 1 (broadcasted dims)
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = np.sum(grad, axis=i, keepdims=True)

    return grad
