import numpy as np

# Define sigmoid function for activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Set random seed for reproducibility
np.random.seed(42)

# Define input matrix
raw_array = []

with open('DataFiles/L30fft16.out', 'r') as file:
	for line in file:
		row = [item for item  in line.split(' ')]
		raw_array.append(row)

numRows = int(raw_array[0][0])
numCols = int(raw_array[0][1])

X = np.zeros(shape=(numRows, numCols))
labels = np.zeros(shape=(1, numRows))
temp = []

i = -1

for row in range (len(raw_array)):
	j = 0
	for col in range (len(raw_array[1])-1): 
		if(row > 0 and col == 0):
			temp.append(int(raw_array[row][col]))

		if(row > 0 and col > 0):
			if(raw_array[row][col] != ''):
				X[i,j] =  raw_array[row][col]
				j += 1

	i += 1

labels = np.array(temp)
labels = labels.reshape(numRows, 1)

# Define output matrix
y = np.random.rand(53, 1)

# Define hyperparameters
learning_rate = 0.1
num_epochs = 40000

# Initialize weights with random values
weights1 = 2 * np.random.random((16, 8)) - 1
weights2 = 2 * np.random.random((8, 1)) - 1

# Train the neural network
for i in range(num_epochs):
    # Forward propagation
    layer1 = sigmoid(np.dot(X, weights1))
    layer2 = sigmoid(np.dot(layer1, weights2))

    # Calculate error
    error = y - layer2

    # Backward propagation
    layer2_delta = error * sigmoid_derivative(layer2)
    layer1_error = layer2_delta.dot(weights2.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # Update weights
    weights2 += learning_rate * layer1.T.dot(layer2_delta)
    weights1 += learning_rate * X.T.dot(layer1_delta)

# Predict output for test input
test_input = [[374.35,296.55,62.14,35.16,36.96,28.18,56.53,75.65,78.38,162.61,56.70,30.64,32.97,67.67,104.95,200.48]]
test_output = sigmoid(np.dot(sigmoid(np.dot(test_input, weights1)), weights2))

print("Input: {}".format(test_input))
print("Output: {}".format(test_output))
