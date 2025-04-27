import numpy as np
from remini.tensor import Tensor

class MSELoss:
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Compute the Mean Squared Error (MSE) loss
        loss_data = np.mean((predictions.data - targets.data) ** 2)
        
        # Create the loss tensor
        loss_tensor = Tensor(loss_data, requires_grad=True)
        
        # Compute the gradient of the loss with respect to predictions (dL/dy)
        # Gradient of MSE: dL/dy = 2 * (y_pred - y_true) / N
        grad_data = 2 * (predictions.data - targets.data) / targets.data.shape[0]
        
        # Set the gradient for the loss
        loss_tensor.grad = grad_data
        
        return loss_tensor
