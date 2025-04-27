# remini/losses/mse.py
import numpy as np
from remini.tensor import Tensor

class MSELoss:
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Access the .data attribute for tensor operations
        loss_data = np.mean((predictions.data - targets.data) ** 2)
        return Tensor(loss_data, requires_grad=True)  # Return the loss wrapped in a Tensor
