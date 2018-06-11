import dataset_processing as proc
import sys

def usage():
	print("Incorrect number of arguments")
	print("USAGE: python add_dataset.py <input_file>")
	exit(0)

def main():
	if len(sys.argv) != 2:
		usage()

	input_file = sys.argv[1]

	dataset = proc.read_file("dataset.csv")
	print("Dataset lines before:", len(dataset))

	new_data = proc.read_file(input_file)
	print("New file lines before:", len(new_data))

	new_lines = proc.remove_duplicate(new_data, dataset)
	print("Added", len(new_lines), "lines")

	proc.create_file(new_lines)

if __name__ == "__main__":
	main()
