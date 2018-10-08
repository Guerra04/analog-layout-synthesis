import pandas as pd
import numpy as np

import dataset_processing as dp
import dataset_dependent as dd

N_DEVICES = 12

def main():
	df = pd.read_csv('DATASETS/dataset_multi_template.csv')

	N_TEMPLATES = 8
	N_DEVICES = 12
	lines = df.shape[0]

	count = np.zeros(N_TEMPLATES)
	wasted_areas = np.zeros([lines, N_TEMPLATES])
	heights = np.zeros([lines, N_TEMPLATES])
	widths = np.zeros([lines, N_TEMPLATES])

	for i in range(N_TEMPLATES):
		if i == 0:
			column_wa = 'wastedarea'
			column_he = 'layoutheight'
			column_wi = 'layoutwidth'
		else:
			column_wa = 'wastedarea.' + str(i)
			column_he = 'layoutheight.' + str(i)
			column_wi = 'layoutwidth.' + str(i)

		wasted_areas[:, i] = df[column_wa]
		heights[:,i] = df[column_he]
		widths[:,i] = df[column_wi]

	aspect_ratio = np.divide(heights, widths)

	n_inputs = 42
	n_outputs_metric = N_DEVICES * 2 + 5 #areas, height, width and template_id
	n_outputs = 3 * n_outputs_metric
	n_cols = n_inputs + n_outputs
	data = np.zeros([lines, n_cols])
	for line in range(lines):
		for col in range(n_inputs):
			data[line, col] = df.iloc[line, col]

		min_area = 99999999999
		min_ar = 99999999999
		max_ar = 0
		#find best templates
		for col in range(N_TEMPLATES):
			if wasted_areas[line, col] < min_area:
				best_template_area = col
				min_area = wasted_areas[line, col]

			if aspect_ratio[line, col] < min_ar:
				best_template_min_ar = col
				min_ar = aspect_ratio[line, col]

			if aspect_ratio[line, col] > max_ar:
				best_template_max_ar = col
				max_ar = aspect_ratio[line, col]

		# MINIMUM WASTED AREA
		metric = 0
		begin = n_inputs + (metric * n_outputs_metric)
		for col in range(n_outputs_metric - 1):
			idx = n_inputs + ((n_outputs_metric - 1) * best_template_area) + col
			data[line, begin + col] = df.iloc[line, idx]
		else:
			data[line, begin + n_outputs_metric-1] = best_template_area

		# MINIMUM ASPECT RATIO
		metric = 1
		begin = n_inputs + (metric * n_outputs_metric)
		for col in range(n_outputs_metric - 1):
			idx = n_inputs + ((n_outputs_metric - 1) * best_template_min_ar) + col
			data[line, begin + col] = df.iloc[line, idx]
		else:
			data[line, begin + n_outputs_metric-1] = best_template_min_ar

		# MAXIMUM ASPECT RATIO
		metric = 2
		begin = n_inputs + (metric * n_outputs_metric)
		for col in range(n_outputs_metric - 1):
			idx = n_inputs + ((n_outputs_metric - 1) * best_template_max_ar) + col
			data[line, begin + col] = df.iloc[line, idx]
		else:
			data[line, begin + n_outputs_metric-1] = best_template_max_ar

	#write to file
	print(data.shape)
	data = data.tolist()
	file = 'DATASETS/dataset_' + str(N_TEMPLATES) + 'best_template_metrics.csv'
	dp.create_file(data, file)

if __name__ == '__main__':
	main()

def center_data(inputs, outputs):
	outputs_center = np.zeros(outputs.shape)

	for line in range(len(outputs)):
		axis_area = (outputs[line, 6*0] + outputs[line, 6*3] + inputs[line, 18])/2
		axis_min_ar = (outputs[line, 2+6*0] + outputs[line, 2+6*3] + inputs[line, 18])/2
		axis_max_ar = (outputs[line, 4+6*0] + outputs[line, 4+6*3] + inputs[line, 18])/2

		for col in range(len(outputs[0,:])):
			outputs_center[line, col] = outputs[line, col]

		for n in range(N_DEVICES):
			outputs_center[line, 0+6*n] = outputs[line, 0+6*n] - axis_area
			outputs_center[line, 2+6*n] = outputs[line, 2+6*n] - axis_min_ar
			outputs_center[line, 4+6*n] = outputs[line, 4+6*n] - axis_max_ar

	return outputs_center

def get_vector(data, feature):
	return_vector = []

	if feature == "w":
		return_vector.append([row[5] for row in data])
		return_vector.append([row[4] for row in data])
		return_vector.append([row[4] for row in data])
		return_vector.append([row[5] for row in data])
		return_vector.append([row[2] for row in data])
		return_vector.append([row[2] for row in data])
		return_vector.append([row[1] for row in data])
		return_vector.append([row[1] for row in data])
		return_vector.append([row[0] for row in data])
		return_vector.append([row[0] for row in data])
		return_vector.append([row[3] for row in data])
		return_vector.append([row[3] for row in data])
		return return_vector

	elif feature == "nf":
		return_vector.append([row[11] for row in data])
		return_vector.append([row[10] for row in data])
		return_vector.append([row[10] for row in data])
		return_vector.append([row[11] for row in data])
		return_vector.append([row[8] for row in data])
		return_vector.append([row[8] for row in data])
		return_vector.append([row[7] for row in data])
		return_vector.append([row[7] for row in data])
		return_vector.append([row[6] for row in data])
		return_vector.append([row[6] for row in data])
		return_vector.append([row[9] for row in data])
		return_vector.append([row[9] for row in data])
		return return_vector

	elif feature == "l":
		return_vector.append([row[17] for row in data])
		return_vector.append([row[16] for row in data])
		return_vector.append([row[16] for row in data])
		return_vector.append([row[17] for row in data])
		return_vector.append([row[14] for row in data])
		return_vector.append([row[14] for row in data])
		return_vector.append([row[13] for row in data])
		return_vector.append([row[13] for row in data])
		return_vector.append([row[12] for row in data])
		return_vector.append([row[12] for row in data])
		return_vector.append([row[15] for row in data])
		return_vector.append([row[15] for row in data])
		return return_vector

	elif feature == "wt":
		return_vector.append([row[18] for row in data])
		return_vector.append([row[20] for row in data])
		return_vector.append([row[22] for row in data])
		return_vector.append([row[24] for row in data])
		return_vector.append([row[26] for row in data])
		return_vector.append([row[28] for row in data])
		return_vector.append([row[30] for row in data])
		return_vector.append([row[32] for row in data])
		return_vector.append([row[34] for row in data])
		return_vector.append([row[36] for row in data])
		return_vector.append([row[38] for row in data])
		return_vector.append([row[40] for row in data])
		return return_vector

	elif feature == "ht":
		return_vector.append([row[19] for row in data])
		return_vector.append([row[21] for row in data])
		return_vector.append([row[23] for row in data])
		return_vector.append([row[25] for row in data])
		return_vector.append([row[27] for row in data])
		return_vector.append([row[29] for row in data])
		return_vector.append([row[31] for row in data])
		return_vector.append([row[33] for row in data])
		return_vector.append([row[35] for row in data])
		return_vector.append([row[37] for row in data])
		return_vector.append([row[39] for row in data])
		return_vector.append([row[41] for row in data])
		return return_vector

	elif feature == "x_area":
		return_vector.append([row[42] for row in data])
		return_vector.append([row[44] for row in data])
		return_vector.append([row[46] for row in data])
		return_vector.append([row[48] for row in data])
		return_vector.append([row[50] for row in data])
		return_vector.append([row[52] for row in data])
		return_vector.append([row[54] for row in data])
		return_vector.append([row[56] for row in data])
		return_vector.append([row[58] for row in data])
		return_vector.append([row[60] for row in data])
		return_vector.append([row[62] for row in data])
		return_vector.append([row[64] for row in data])
		return return_vector

	elif feature == "y_area":
		return_vector.append([row[43] for row in data])
		return_vector.append([row[45] for row in data])
		return_vector.append([row[47] for row in data])
		return_vector.append([row[49] for row in data])
		return_vector.append([row[51] for row in data])
		return_vector.append([row[53] for row in data])
		return_vector.append([row[55] for row in data])
		return_vector.append([row[57] for row in data])
		return_vector.append([row[59] for row in data])
		return_vector.append([row[61] for row in data])
		return_vector.append([row[63] for row in data])
		return_vector.append([row[65] for row in data])
		return return_vector

	elif feature == "x_min_ar":
		return_vector.append([row[71] for row in data])
		return_vector.append([row[73] for row in data])
		return_vector.append([row[75] for row in data])
		return_vector.append([row[77] for row in data])
		return_vector.append([row[79] for row in data])
		return_vector.append([row[81] for row in data])
		return_vector.append([row[83] for row in data])
		return_vector.append([row[85] for row in data])
		return_vector.append([row[87] for row in data])
		return_vector.append([row[89] for row in data])
		return_vector.append([row[91] for row in data])
		return_vector.append([row[93] for row in data])
		return return_vector

	elif feature == "y_min_ar":
		return_vector.append([row[72] for row in data])
		return_vector.append([row[74] for row in data])
		return_vector.append([row[76] for row in data])
		return_vector.append([row[78] for row in data])
		return_vector.append([row[80] for row in data])
		return_vector.append([row[82] for row in data])
		return_vector.append([row[84] for row in data])
		return_vector.append([row[86] for row in data])
		return_vector.append([row[88] for row in data])
		return_vector.append([row[90] for row in data])
		return_vector.append([row[92] for row in data])
		return_vector.append([row[94] for row in data])
		return return_vector

	elif feature == "x_max_ar":
		return_vector.append([row[100] for row in data])
		return_vector.append([row[102] for row in data])
		return_vector.append([row[104] for row in data])
		return_vector.append([row[106] for row in data])
		return_vector.append([row[108] for row in data])
		return_vector.append([row[110] for row in data])
		return_vector.append([row[112] for row in data])
		return_vector.append([row[114] for row in data])
		return_vector.append([row[116] for row in data])
		return_vector.append([row[118] for row in data])
		return_vector.append([row[120] for row in data])
		return_vector.append([row[122] for row in data])
		return return_vector

	elif feature == "y_max_ar":
		return_vector.append([row[101] for row in data])
		return_vector.append([row[103] for row in data])
		return_vector.append([row[105] for row in data])
		return_vector.append([row[107] for row in data])
		return_vector.append([row[109] for row in data])
		return_vector.append([row[111] for row in data])
		return_vector.append([row[113] for row in data])
		return_vector.append([row[115] for row in data])
		return_vector.append([row[117] for row in data])
		return_vector.append([row[119] for row in data])
		return_vector.append([row[121] for row in data])
		return_vector.append([row[123] for row in data])
		return return_vector

	elif feature == "metrics_area":
		return_vector.append([row[66] for row in data])
		return_vector.append([row[67] for row in data])
		return_vector.append([row[68] for row in data])
		return_vector.append([row[69] for row in data])
		return_vector.append([row[70] for row in data])
		return return_vector

	elif feature == "metrics_min_ar":
		return_vector.append([row[95] for row in data])
		return_vector.append([row[96] for row in data])
		return_vector.append([row[97] for row in data])
		return_vector.append([row[98] for row in data])
		return_vector.append([row[99] for row in data])
		return return_vector

	elif feature == "metrics_max_ar":
		return_vector.append([row[124] for row in data])
		return_vector.append([row[125] for row in data])
		return_vector.append([row[126] for row in data])
		return_vector.append([row[127] for row in data])
		return_vector.append([row[128] for row in data])
		return return_vector

	else:
		return False

def get_inputs(data):
	data = dd.to_float(data)

	w = get_vector(data, "w")
	l = get_vector(data, "l")
	nf = get_vector(data, "nf")
	wt = get_vector(data, "wt")
	ht = get_vector(data, "ht")

	inputs = []
	inputs.append(w)
	inputs.append(l)
	inputs.append(nf)
	inputs.append(wt)
	inputs.append(ht)
	n_inputs = 5 * N_DEVICES
	inputs_ = np.asarray(inputs)
	inputs = inputs_
	inputs = inputs.transpose() #puts different lines along 1st dimension
	inputs = np.reshape(inputs, (-1, n_inputs)) #reduce to 2D

	return inputs

def get_outputs(data):
	data = dd.to_float(data)

	x_area = get_vector(data, "x_area")
	y_area = get_vector(data, "y_area")
	x_min_ar = get_vector(data, "x_min_ar")
	y_min_ar = get_vector(data, "y_min_ar")
	x_max_ar = get_vector(data, "x_max_ar")
	y_max_ar = get_vector(data, "y_max_ar")

	outputs = []
	outputs.append(x_area)
	outputs.append(y_area)
	outputs.append(x_min_ar)
	outputs.append(y_min_ar)
	outputs.append(x_max_ar)
	outputs.append(y_max_ar)
	n_outputs = 3 * (2 * N_DEVICES)
	outputs_ = np.asarray(outputs)
	outputs = outputs_
	outputs = outputs.transpose()
	outputs = np.reshape(outputs, (-1, n_outputs))

	return outputs

def get_metrics(data):
	data = dd.to_float(data)

	metrics_area = get_vector(data, "metrics_area")
	metrics_min_ar = get_vector(data, "metrics_min_ar")
	metrics_max_ar = get_vector(data, "metrics_max_ar")

	metrics = []
	metrics.append(metrics_area)
	metrics.append(metrics_min_ar)
	metrics.append(metrics_max_ar)
	n_metrics = 3 * 5
	metrics_ = np.asarray(metrics)
	metrics = metrics_
	metrics = metrics.transpose()
	metrics = np.reshape(metrics, (-1, n_metrics))

	return metrics
