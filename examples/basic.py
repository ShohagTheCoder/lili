import numpy as np
from remini.layers.dense import Dense
from remini.optim.sgd import SGD
from remini.tensor import Tensor
from remini.losses.mse_loss import MSELoss

# Input-output data
x_train = Tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = Tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

# Define layers manually
layer1 = Dense(1, 2)   # First layer (1 input -> 4 neurons)
layer2 = Dense(2, 1)    # Second layer (4 -> 1 output)

# Optimizer and loss
optimizer = SGD(params=[layer1.weights, layer1.bias, layer2.weights, layer2.bias], lr=0.01)
loss_fn = MSELoss()

# Training loop
for epoch in range(100):
    optimizer.zero_grad()   # <-- 1. Zero gradients first

    out1 = layer1.forward(x_train)
    y_pred = layer2.forward(out1)
    loss = loss_fn(y_pred, y_train)
    
    # Grad output
    # dL/dy_pred = 2 * (y_pred - y_true) / N
    grad_output = (2 * (y_pred.data - y_train.data)) / y_train.data.shape[0]


    # === Manual backward pass ===
    grad_output = (2 * (y_pred.data - y_train.data)) / y_train.data.shape[0]  # dL/dy_pred
    grad_out2 = layer2.backward(Tensor(grad_output))  # dL/dlayer2 input
    _ = layer1.backward(grad_out2)       # <-- 2. Backward from loss (automatically propagate)
    
    optimizer.step()        # <-- 3. Update parameters

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

# Final output
print("\nTrained Model Output:")
predictions = y_pred.data
targets = y_train.data

print("Input\tPredicted\tTarget")
for x, pred, target in zip(x_train.data, predictions, targets):
    print(f"{x[0]:.1f}\t{pred[0]:.4f}\t\t{target[0]:.1f}")
