import matplotlib.pyplot as plt
import matplotlib.patches as patches

import dataset_processing as dp
import dataset_dependent as dd

def draw_rectangle(x, y, w, h, color):
	rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor=color, facecolor='none')
	return rect

def main():
	N_DEVICES = 12
	FEATURE = 1
	data = dp.read_file("dataset.csv")
	data = dd.to_float(data)
	data = dd.normalize(data)

	h = dd.get_vector(data, "ht")
	x = dd.get_vector(data, "x")
	y = dd.get_vector(data, "y")
	w = dd.get_vector(data, "wt")

	print("-------------------Height--------------")
	for n in range(12):
		print(h[n][FEATURE])
	print("-------------------Width--------------")
	for n in range(12):
		print(w[n][FEATURE])
	print("-------------------X--------------")
	for n in range(12):
		print(x[n][0][FEATURE])
	print("-------------------Y--------------")
	for n in range(12):
		print(y[n][0][FEATURE])

	fig,ax = plt.subplots()
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm']
	rect = []
	for n in range(N_DEVICES):
		rect.append(draw_rectangle(x[n][0][FEATURE], y[n][0][FEATURE], w[n][FEATURE], h[n][FEATURE], colors[n]))

	for r in rect:
		ax.add_patch(r)

	plt.show()

if __name__ == "__main__":
	main()
