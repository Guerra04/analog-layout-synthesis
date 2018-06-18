from HYPERPARAMS import *
import numpy as np
import pandas as pd

class Cell(object):
	def __init__(self, x, y, width, height, id):
		self.center_x = x + width/2
		self.center_y = y + height/2
		self.max_x = x + width
		self.min_x = x
		self.max_y = y + height
		self.min_y = y
		self.id = id

class Edge(object):
	def __init__(self, id1, id2):
		self.id1 = id1
		self.id2 = id2

def compute_cells(inputs, outputs):
	cells = []
	for n in range(N_DEVICES):
		c = Cell(outputs[2*n], outputs[1+2*n], inputs[3+5*n], inputs[4+5*n], n)
		cells.append(c)
	return cells


def compute_graph_vertical(inputs, outputs):

	cells = compute_cells(inputs, outputs)

	sorted_cells = sorted(cells, key=lambda c: c.center_y, reverse=True)

	edges = []
	levels = []
	th = 1e-7
	while(sorted_cells):
		current_level = []
		break_point = 0
		for c in sorted_cells:
			if abs(c.center_y - sorted_cells[0].center_y) < th:
				break_point += 1
			else:
				break

		for i in range(break_point):
			current_level.append(sorted_cells[i])
		for i in range(break_point):
			sorted_cells.pop(0)

		levels.append(current_level)

	for l1 in range(len(levels)-1, -1, -1):
		for c1 in levels[l1]:
			found = False
			for l2 in range(l1-1, -1, -1):
				if(found):
					break
				for c2 in levels[l2]:
					if min(c1.max_x, c2.max_x) > max(c1.min_x, c2.min_x):
						found = True
						e = Edge(c2.id, c1.id)
						edges.append(e)
						break
			else:
				e = Edge(None, c1.id)
				edges.append(e)

	return edges

def compute_graph_horizontal(inputs, outputs):

	cells = compute_cells(inputs, outputs)

	sorted_cells = sorted(cells, key=lambda c: c.center_x, reverse=True)

	edges = []
	levels = []
	th = 1e-7
	while(sorted_cells):
		current_level = []
		break_point = 0
		for c in sorted_cells:
			if abs(c.center_x - sorted_cells[0].center_x) < th:
				break_point += 1
			else:
				break

		for i in range(break_point):
			current_level.append(sorted_cells[i])
		for i in range(break_point):
			sorted_cells.pop(0)

		levels.append(current_level)

	for l1 in range(len(levels)-1, -1, -1):
		for c1 in levels[l1]:
			found = False
			for l2 in range(l1-1, -1, -1):
				if(found):
					break
				for c2 in levels[l2]:
					if min(c1.max_y, c2.max_y) > max(c1.min_y, c2.min_y):
						found = True
						e = Edge(c2.id, c1.id)
						edges.append(e)
						break
			else:
				e = Edge(None, c1.id)
				edges.append(e)

	return edges

def compute_similarity(graph_predict, graph_template):

	count = 0
	for e1 in graph_predict:
		for e2 in graph_template:
			if e1.id1 == e2.id1 and e1.id2 == e2.id2:
				count += 1

	return count

def read_all_templates():

	df = pd.read_csv('DATASETS/dataset_multi_template.csv')

	lines = df.shape[0]
	n_inputs = 42
	n_outputs_template = 2*N_DEVICES + 4

	outputs = np.zeros((lines, N_TEMPLATES, 2*N_DEVICES))

	for line in range(lines):
		for temp in range(N_TEMPLATES):
			for n in range(N_DEVICES):
				col = idx = n_inputs + ((n_outputs_template) * temp) + 2*n
				outputs[line, temp, 0+2*n] = df.iloc[line, col]

				col = idx = n_inputs + ((n_outputs_template) * temp) + 2*n + 1
				outputs[line, temp, 1+2*n] = df.iloc[line, col]

	return outputs

def compute_similarity_matrix(inputs, predict, outputs_templates):

	similarities = np.zeros((N_TEMPLATES,1))

	for temp in range(N_TEMPLATES):
		outputs = outputs_templates[temp, :]

		graph_predict_vertical = compute_graph_vertical(inputs, predict)
		graph_predict_horizontal = compute_graph_horizontal(inputs, predict)

		graph_template_vertical = compute_graph_vertical(inputs, outputs)
		graph_template_horizontal = compute_graph_horizontal(inputs, outputs)

		sim_vertical = compute_similarity(graph_predict_vertical, graph_template_vertical)
		sim_horizontal = compute_similarity(graph_predict_horizontal, graph_template_horizontal)

		similarity = (sim_vertical + sim_horizontal)/(len(graph_predict_vertical) + len(graph_predict_horizontal))

		similarity = similarity * 100

		similarities[temp] = similarity

	return similarities

def normalize(similarity_matrix):


	return
