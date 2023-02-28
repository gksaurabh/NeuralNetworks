import numpy as np

raw_array = []
with open('DataFiles/L30fft16.out', 'r') as file:
	for line in file:
		row = [item for item  in line.split(' ')]
		raw_array.append(row)




input_array = np.zeros(shape=((int(raw_array[0][0])),(int(raw_array[0][1])) ))
labels = np.zeros(shape=(int(raw_array[0][0]),1))

i = -1


for row in range (len(raw_array)):
	j = 0
	for col in range (len(raw_array[1])-1): 
		if(row > 0 and col > 0):
			input_array[i,j] =  raw_array[row][col]
			j += 1

	i += 1

print(input_array)

