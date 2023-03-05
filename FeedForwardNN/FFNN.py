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
			if(raw_array[row][col] != ''):
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
	return 1/(1+np.exp(-x))

def sigdr(x):
	return sigmoid(x) * (1 - sigmoid(x))

for epoch in range(50000):
	inputs = input_matrix
	XW = np.dot(inputs, weights) + bias
	z = sigmoid(XW)
	error = z - labels
	print("Training: ",error.sum())
	dcost = error
	dpred = sigdr(z)
	z_del = dcost * dpred
	inputs = input_matrix.T
	weights = weights - learning_rate*np.dot(inputs, z_del)

	for num in z_del:
		bias = bias - learning_rate*num

single_ot = np.array([14.62,21.53,1.97,0.84,0.52,0.51,0.42,0.30,0.53,2.13,2.82,0.89,1.55,1.50,0.91,0.96,0.93,1.47,1.68,0.94,1.25,0.59,1.00,0.53,0.84,0.45,0.39,0.31,0.31,0.33,0.49,0.32,0.65,0.43,0.38,0.22,0.25,0.30,0.37,0.50,0.55,0.57,0.52,0.43,0.35,0.28,0.26,0.23,0.25,0.27,0.24,0.28,0.30,0.45,0.41,0.50,0.45,0.52,0.60,0.63,0.59,0.74,0.79,0.86,0.83,1.01,1.24,1.53,1.67,1.64,1.37,1.38,1.11,1.01,0.91,0.90,0.99,0.93,0.88,0.87,1.28,1.10,0.44,0.93,0.81,1.58,1.92,2.28,2.38,2.83,3.13,3.56,2.57,2.06,1.41,1.04,0.98,0.77,0.72,0.70,0.59,0.58,0.56,0.53,0.53,0.49,0.48,0.45,0.38,0.37,0.36,0.32,0.31,0.29,0.30,0.30,0.32,0.36,0.40,0.41,0.42,0.40,0.49,0.60,0.62,0.69,0.73,1.02,1.12,1.19,1.11,1.05,1.05,1.15,1.19,1.18,1.23,1.13,1.11,1.22,1.45,1.77,2.22,2.64,2.79,2.85,2.54,2.58,2.41,2.31])
result = sigmoid(np.dot(single_ot,weights) + bias)

summation = 0
for val in result:
	summation = summation + val

probability = summation/(len(result) + 1)

print("RESULT = ", result)
print("PROBABILITY =  ", probability)	
