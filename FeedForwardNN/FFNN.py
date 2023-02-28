import numpy as np

raw_array = []

with open('DataFiles/L30fft16.out', 'r') as file:
	for line in file:
		row = [item for item  in line.split(' ')]
		raw_array.append(row)

numRows = int(raw_array[0][0])
numCols = int(raw_array[0][1])

input_matrix = np.zeros(shape=(numRows, numCols))
labels = np.zeros(shape=(1, numRows))
temp = []

i = -1

for row in range (len(raw_array)):
	j = 0
	for col in range (len(raw_array[1])-1): 
		if(row > 0 and col == 0):
			temp.append(int(raw_array[row][col]))

		if(row > 0 and col > 0):
			input_matrix[i,j] =  raw_array[row][col]
			j += 1

	i += 1

labels = np.array(temp)
labels = labels.reshape(numRows, 1)

print(labels)

np.random.seed(51)
weights = np.random.rand(numCols , numRows)
bias = np.random.rand(1)
learning_rate = 0.05

def sigmoid(x):
	return (1/(1+np.exp(-x)))

def sigdr(x):
	return sigmoid(x) * (1 - sigmoid(x))

for epoch in range(100000):
	inputs = input_matrix
	XW = np.dot(inputs, weights) + bias
	z = sigmoid(XW)
	error = z - labels
	print(error.sum())
	dcost = error
	dpred = sigdr(z)
	z_del = dcost * dpred
	inputs = input_matrix.T
	weights = weights - learning_rate*np.dot(inputs, z_del)

	for num in z_del:
		bias = bias - learning_rate*num

