import sys
import numpy as np

def read_file(input_file):
	print("READING file...")
	data = []
	with open(input_file) as file:
		for line in file:
			if line[0] != '_': #skip first line
				lines = line.split(",")
				data.append(lines)
			else:
				first_line = line
	print("FINISHED reading")
	return data, first_line

def remove_duplicate(data, new_data = []):
	print("REMOVING duplicates...")
	return_data = []
	same = False
	for i in range(len(data)):
		for j in range(len(new_data)):
			same = False
			for col in range(42): #ninputs
				if data[i][col] != new_data[j][col]:
					break
			else:
				same = True
			if same:
				break
		else:
			new_data.append(data[i])
			return_data.append(data[i])
	print("FINISHED removing duplicates")
	return return_data

def create_file(data, output_file, header):
	print('WRITING', output_file, '...')
	file = open(output_file, "a+")
	file.write(header)
	for d in data:
		for col in range(len(d)):
			file.write(d[col])
			if col < (len(d)-1):
				file.write(",")
		#file.write("\n")
	file.close()
	print("FINISHED writing file")
	return

def reshape(data, new_dim):
	num_examples = data.shape[0]
	n_devices = data.shape[1]
	n_inputs = data.shape[2]
	reshaped = np.zeros((num_examples, new_dim))

	for i in range(num_examples):
		for j in range(n_devices): #ndevices
			for k in range(n_inputs):
				reshaped[i,j*n_inputs+k] = data[i,j,k]
	return reshaped

#scale data

#get input, outputs separado


def usage():
	print("Incorrect number of arguments")
	print("USAGE: python dataset_processing.py <input_file> <output_file>")
	exit(0)
#-------------------------------------MAIN--------------------------------------
def main():
	if len(sys.argv) != 3:
		usage()

	input_file = sys.argv[1]
	output_file = sys.argv[2]

	data, first_line = read_file(input_file)
	print("Lines before:", len(data))
	new_data = remove_duplicate(data)
	print("Lines after:", len(new_data))
	create_file(new_data, output_file, first_line)

if __name__ == "__main__":
	main()
