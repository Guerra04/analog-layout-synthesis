import math
import random
import sys
import dataset_processing as dp
import dataset_dependent as dd

def usage():
	print("Incorrect number of arguments")
	print("USAGE: python separate_train_test.py <input_file>")
	exit(0)

def main():

	if len(sys.argv) != 2:
		usage()

	input_file = sys.argv[1]

	data = dp.read_file(input_file)

	train_percentage = 0.8;

	num_examples = len(data)
	n_train = math.ceil(train_percentage * num_examples)
	n_test = num_examples-n_train

	indexes = list(range(0, num_examples))
	random.shuffle(indexes)

	data_train = []
	data_test = []
	for i in range(len(indexes)):
		if i < n_train:
			data_train.append(data[indexes[i]])
		else:
			data_test.append(data[indexes[i]])

	print('TRAIN:', len(data_train))
	print('TEST:', len(data_test))

	input_file = input_file.replace('.csv', '')
	output_file = input_file + '_train.csv'
	dp.create_file(data_train, output_file)
	output_file = input_file + '_test.csv'
	dp.create_file(data_test, output_file)

if __name__ == "__main__":
	main()
