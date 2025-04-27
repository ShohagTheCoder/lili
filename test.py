import numpy as np
from remini.layers.dense import Dense
from remini.optim.sgd import SGD
from remini.tensor import Tensor
from remini.losses.mse_loss import MSELoss

# Input-output data
x_train = Tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = Tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

# Define layers manually
layers = [
    Dense(1, 16),   # First layer (1 input -> 4 neurons)
    Dense(16, 1)    # Second layer (4 -> 1 output)
]

# Collect parameters from all layers
params = []
for layer in layers:
    params.extend(layer.parameters())

# Optimizer and loss
optimizer = SGD(params=params, lr=0.01)
loss_fn = MSELoss()

# Training loop
for epoch in range(2000):
    out = x_train
    for layer in layers:
        out = layer(out)  # Forward pass through each layer

    y_pred = out
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.grad = np.ones_like(loss.data)
    y_pred.grad = loss.grad
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
        
# Final output
print("\nTrained Model Output:")
predictions = y_pred.data
targets = y_train.data

print("Input\tPredicted\tTarget")
for x, pred, target in zip(x_train.data, predictions, targets):
    print(f"{x[0]:.1f}\t{pred[0]:.4f}\t\t{target[0]:.1f}")
