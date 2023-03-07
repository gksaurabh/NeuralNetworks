import numpy as np
import matplotlib.pyplot as pyp
import math

# Define sigmoid function for activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Set random seed for reproducibility
np.random.seed(42)


# Method to read files for training
def readFile(filename):
    # Define input matrix
    labels = []
    X = []
    with open(filename, 'r') as file:
        row_counter = 0
        for line in file:
            line = line.replace('  ', ' ')
            # Getting number of rows and column from the first line of the file
            if(row_counter == 0):
                numRows = int(line.split(' ')[0])
                numCols = int(line.split(' ')[1])
                row_counter += 1
            # Getting the data matrix
            else:
                labels.append(int(line.split(' ')[0]))
                i = 1
                temp = []
                for i in range(numCols):
                    if(line.split(' ')[i] != ''):
                        temp.append(float(line.split(' ')[i]))
                X.append(temp)
            row_counter += 1


    X = np.array(X)
    labels = np.array(labels)
    labels = labels.reshape(numRows, 1)
    return_array = [numRows, numCols, X, labels]
    return return_array



# Method to train ANN
def trainer(output, epochs):
    # Define variables for graph
    error_array = []
    epoch_array = []
    numRows = output[0]
    numCols = output[1]
    X = output[2]
    labels = output[3]
    # Define output matrix
    y = np.random.rand(53, 1)

    # Define hyperparameters
    learning_rate = 0.1
    num_epochs = epochs

    # Initialize weights with random values
    weights1 = 2 * np.random.random((numCols, 8)) - 1
    weights2 = 2 * np.random.random((8, numCols)) - 1


    # Creating graph for tracking errors
    pyp.xlabel('Epoch')
    pyp.ylabel('Error')
    pyp.title("Error Range Graph")

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

        # add errors to an array for graphs
        error_array.append(math.cos(error.sum()))
        epoch_array.append(i)

    pyp.plot(epoch_array, error_array)
    pyp.show()
    


list_of_files = ["L30fft16.out" , "L30fft150.out" , "L30fft1000.out" , "L30fft25.out" , "L30fft_32.out" , "L30fft_64.out"];
filePath = "./DataFiles/"
for file in list_of_files:
    print("Reading: ", file)
    output = readFile(filePath + file)
    trainer(output, 40000)