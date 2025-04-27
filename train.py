import numpy as np
from remini.layers.dense import Dense
from remini.models.sequential import Sequential
from remini.optim.adam import Adam
from remini.tensor import Tensor
from remini.losses.mse_loss import MSELoss

# Input-output data for y = 2x^3 - 5x^2 + 3x + 8
x_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = np.array([[2.0], [8.0], [38.0], [118.0], [258.0]])

# Standardize x_train
x_mean = np.mean(x_train)
x_std = np.std(x_train)
x_train_standardized = (x_train - x_mean) / x_std

# Convert numpy arrays to Tensor
x_train_tensor = Tensor(x_train_standardized)
y_train_tensor = Tensor(y_train)

# Define the model using Sequential API
model = Sequential(
    Dense(1, 64),  # First Dense layer
    Dense(64, 64),  # Second Dense layer
    Dense(64, 1)    # Output layer
)

# Optimizer and loss function
optimizer = Adam(params=model.parameters(), lr=0.0001)
loss_fn = MSELoss()

# Training loop
epochs = 50000
for epoch in range(epochs):
    # Zero gradients from previous step
    optimizer.zero_grad()

    # Forward pass: Get model predictions
    y_pred = model.forward(x_train_tensor)

    # Calculate loss
    loss = loss_fn(y_pred, y_train_tensor)

    # Compute gradients via backward pass
    grad_output = (2 * (y_pred.data - y_train_tensor.data)) / y_train_tensor.data.shape[0]
    grad = Tensor(grad_output)

    # Backward pass through each layer (reversed)
    for layer in reversed(model.layers):
        grad = layer.backward(grad)

    # Optimizer step: Update parameters
    optimizer.step()

    # Print loss every 200 epochs
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

# Denormalize x_train for final output
x_original = (x_train_tensor.data * x_std + x_mean)

# Display predictions
print("\nTrained Model Output:")
predictions = y_pred.data
targets = y_train_tensor.data

print("Input\tPredicted\tTarget")
for x, pred, target in zip(x_original, predictions, targets):
    print(f"{x[0]:.1f}\t{pred[0]:.4f}\t\t{target[0]:.1f}")
