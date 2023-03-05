import numpy as np

# Define input matrix X with 3 examples, each with 2 features
X = np.array([[1, 2], [3, 4], [5, 6]])

# Define output matrix y with 3 examples, each with 1 output
y = np.array([[0], [1], [0]])

# Define hyperparameters
input_layer_size = 2
hidden_layer_size = 3
output_layer_size = 1
learning_rate = 0.1

# Initialize weights and biases for the hidden layer and output layer
W1 = np.random.randn(input_layer_size, hidden_layer_size)
b1 = np.zeros((1, hidden_layer_size))
W2 = np.random.randn(hidden_layer_size, output_layer_size)
b2 = np.zeros((1, output_layer_size))

# Define sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define feedforward function
def feedforward(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)
    return y_hat

# Define cost function
def cost(y, y_hat):
    return np.mean(-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

# Train the model using backpropagation
for i in range(1000):
    # Forward pass
    y_hat = feedforward(X, W1, b1, W2, b2)
    
    # Backward pass
    dL_dy_hat = (y_hat - y) / len(X)
    dL_dz2 = dL_dy_hat * y_hat * (1 - y_hat)
    dL_da1 = np.dot(dL_dz2, W2.T)
    dL_dz1 = dL_da1 * sigmoid(z1) * (1 - sigmoid(z1))
    dW2 = np.dot(a1.T, dL_dz2)
    db2 = np.sum(dL_dz2, axis=0, keepdims=True)
    dW1 = np.dot(X.T, dL_dz1)
    db1 = np.sum(dL_dz1, axis=0, keepdims=True)
    
    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# Make predictions on new input
X_new = np.array([[2, 3], [4, 5]])
y_pred = feedforward(X_new, W1, b1, W2, b2)
print(y_pred)
