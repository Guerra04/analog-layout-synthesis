import numpy as np

N_DEVICES = 12

def to_float(data):
	new_data = []
	for d in range(len(data)):
		new_data.append([])
		for col in data[d]:
			new_data[d].append(float(col))

	return new_data

def normalize(data):
	new_data = []
	for d in range(len(data)):
		new_data.append([])
		for col in data[d]:
			new_col = col/5e-9
			new_data[d].append(float(new_col))
	return new_data

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

	elif feature == "x":

		for n in range(12):
			return_vector.append([])

		#template A
		return_vector[0].append([row[42] for row in data])
		return_vector[1].append([row[44] for row in data])
		return_vector[2].append([row[46] for row in data])
		return_vector[3].append([row[48] for row in data])
		return_vector[4].append([row[50] for row in data])
		return_vector[5].append([row[52] for row in data])
		return_vector[6].append([row[54] for row in data])
		return_vector[7].append([row[56] for row in data])
		return_vector[8].append([row[58] for row in data])
		return_vector[9].append([row[60] for row in data])
		return_vector[10].append([row[62] for row in data])
		return_vector[11].append([row[64] for row in data])
		#template B
		return_vector[0].append([row[66] for row in data])
		return_vector[1].append([row[68] for row in data])
		return_vector[2].append([row[70] for row in data])
		return_vector[3].append([row[72] for row in data])
		return_vector[4].append([row[74] for row in data])
		return_vector[5].append([row[76] for row in data])
		return_vector[6].append([row[78] for row in data])
		return_vector[7].append([row[80] for row in data])
		return_vector[8].append([row[82] for row in data])
		return_vector[9].append([row[84] for row in data])
		return_vector[10].append([row[86] for row in data])
		return_vector[11].append([row[88] for row in data])
		# template C
		return_vector[0].append([row[90] for row in data])
		return_vector[1].append([row[92] for row in data])
		return_vector[2].append([row[94] for row in data])
		return_vector[3].append([row[96] for row in data])
		return_vector[4].append([row[98] for row in data])
		return_vector[5].append([row[100] for row in data])
		return_vector[6].append([row[102] for row in data])
		return_vector[7].append([row[104] for row in data])
		return_vector[8].append([row[106] for row in data])
		return_vector[9].append([row[108] for row in data])
		return_vector[10].append([row[110] for row in data])
		return_vector[11].append([row[112] for row in data])
		return return_vector

	elif feature == "y":

		for n in range(12):
			return_vector.append([])

		#template A
		return_vector[0].append([row[43] for row in data])
		return_vector[1].append([row[45] for row in data])
		return_vector[2].append([row[47] for row in data])
		return_vector[3].append([row[49] for row in data])
		return_vector[4].append([row[51] for row in data])
		return_vector[5].append([row[53] for row in data])
		return_vector[6].append([row[55] for row in data])
		return_vector[7].append([row[57] for row in data])
		return_vector[8].append([row[59] for row in data])
		return_vector[9].append([row[61] for row in data])
		return_vector[10].append([row[63] for row in data])
		return_vector[11].append([row[65] for row in data])
		#template B
		return_vector[0].append([row[67] for row in data])
		return_vector[1].append([row[69] for row in data])
		return_vector[2].append([row[71] for row in data])
		return_vector[3].append([row[73] for row in data])
		return_vector[4].append([row[75] for row in data])
		return_vector[5].append([row[77] for row in data])
		return_vector[6].append([row[79] for row in data])
		return_vector[7].append([row[81] for row in data])
		return_vector[8].append([row[83] for row in data])
		return_vector[9].append([row[85] for row in data])
		return_vector[10].append([row[87] for row in data])
		return_vector[11].append([row[89] for row in data])
		# template C
		return_vector[0].append([row[91] for row in data])
		return_vector[1].append([row[93] for row in data])
		return_vector[2].append([row[95] for row in data])
		return_vector[3].append([row[97] for row in data])
		return_vector[4].append([row[99] for row in data])
		return_vector[5].append([row[101] for row in data])
		return_vector[6].append([row[103] for row in data])
		return_vector[7].append([row[105] for row in data])
		return_vector[8].append([row[107] for row in data])
		return_vector[9].append([row[109] for row in data])
		return_vector[10].append([row[111] for row in data])
		return_vector[11].append([row[113] for row in data])
		return return_vector

	else:
		return False

def get_inputs(data):
	data = to_float(data)
	#data = normalize(data)

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

	#scaler_in = StandardScaler()
	#scaler_in.fit(inputs)
	#inputs = scaler_in.transform(inputs)

	return inputs

def get_outputs(data, template):
	data = to_float(data)
	#data = normalize(data)

	x = get_vector(data, "x")
	y = get_vector(data, "y")

	outputs = []
	x_ = np.asarray(x)
	y_ = np.asarray(y)
	outputs.append(x_[:,template,:])
	outputs.append(y_[:,template,:])
	n_outputs = 2 * N_DEVICES
	outputs_ = np.asarray(outputs)
	outputs = outputs_
	outputs = outputs.transpose()
	outputs = np.reshape(outputs, (-1, n_outputs))

	#scaler_out = StandardScaler()
	#scaler_out.fit(outputs)
	#outputs = scaler_out.transform(outputs)

	return outputs

def get_half_outputs(outputs_all, inputs):
	outputs = np.zeros([len(outputs_all), N_DEVICES+1])

	for line in range(len(outputs_all)):
		#pair 0-3
		if outputs_all[line, 0] < outputs_all[line, 6]:
			outputs[line, 0] = outputs_all[line, 0]
			outputs[line, 1] = outputs_all[line, 1]
		if outputs_all[line, 0] > outputs_all[line, 6]:
			outputs[line, 0] = outputs_all[line, 6]
			outputs[line, 1] = outputs_all[line, 7]

		#pair 1-2
		if outputs_all[line, 2] < outputs_all[line, 4]:
			outputs[line, 2] = outputs_all[line, 2]
			outputs[line, 3] = outputs_all[line, 3]
		if outputs_all[line, 2] > outputs_all[line, 4]:
			outputs[line, 2] = outputs_all[line, 4]
			outputs[line, 3] = outputs_all[line, 5]

		#pair 4-5
		if outputs_all[line, 8] < outputs_all[line, 10]:
			outputs[line, 4] = outputs_all[line, 8]
			outputs[line, 5] = outputs_all[line, 9]
		if outputs_all[line, 8] > outputs_all[line, 10]:
			outputs[line, 4] = outputs_all[line, 10]
			outputs[line, 5] = outputs_all[line, 11]

		#pair 6-7
		if outputs_all[line, 12] < outputs_all[line, 14]:
			outputs[line, 6] = outputs_all[line, 12]
			outputs[line, 7] = outputs_all[line, 13]
		if outputs_all[line, 12] > outputs_all[line, 14]:
			outputs[line, 6] = outputs_all[line, 14]
			outputs[line, 7] = outputs_all[line, 15]

		#pair 8-9
		if outputs_all[line, 16] < outputs_all[line, 18]:
			outputs[line, 8] = outputs_all[line, 16]
			outputs[line, 9] = outputs_all[line, 17]
		if outputs_all[line, 16] > outputs_all[line, 18]:
			outputs[line, 8] = outputs_all[line, 18]
			outputs[line, 9] = outputs_all[line, 19]

		#pair 10-11
		if outputs_all[line, 20] < outputs_all[line, 22]:
			outputs[line, 10] = outputs_all[line, 20]
			outputs[line, 11] = outputs_all[line, 21]
		if outputs_all[line, 20] > outputs_all[line, 22]:
			outputs[line, 10] = outputs_all[line, 22]
			outputs[line, 11] = outputs_all[line, 23]

		#symmetry axis
		#axis = (x[0] + x[pair(0)] + width[pair(0)])/2
		axis = (outputs_all[line, 0] + outputs_all[line, 6] + inputs[line, 18])/2
		outputs[line, 12] = axis

	return outputs

def get_all_outputs(outputs_half, inputs):
	outputs = np.zeros([len(outputs_half), 2*N_DEVICES])

	for line in range(len(outputs_half)):
		axis = outputs_half[line, 12]

		#pair 0-3
		outputs[line, 0] = outputs_half[line, 0]
		outputs[line, 1] = outputs_half[line, 1]
		outputs[line, 6] = (axis-outputs_half[line,0]) + axis - inputs[line, 18]
		outputs[line, 7] = outputs_half[line, 1]

		#pair 1-2
		outputs[line, 2] = outputs_half[line, 2]
		outputs[line, 3] = outputs_half[line, 3]
		outputs[line, 4] = (axis-outputs_half[line,2]) + axis - inputs[line, 13]
		outputs[line, 5] = outputs_half[line, 3]

		#pair 4-5
		outputs[line, 8] = outputs_half[line, 4]
		outputs[line, 9] = outputs_half[line, 5]
		outputs[line, 10] = (axis-outputs_half[line,4]) + axis - inputs[line, 28]
		outputs[line, 11] = outputs_half[line, 5]

		#pair 6-7
		outputs[line, 12] = outputs_half[line, 6]
		outputs[line, 13] = outputs_half[line, 7]
		outputs[line, 14] = (axis-outputs_half[line,6]) + axis - inputs[line, 38]
		outputs[line, 15] = outputs_half[line, 7]

		#pair 8-9
		outputs[line, 16] = outputs_half[line, 8]
		outputs[line, 17] = outputs_half[line, 9]
		outputs[line, 18] = (axis-outputs_half[line,8]) + axis - inputs[line, 48]
		outputs[line, 19] = outputs_half[line, 9]

		#pair 10-11
		outputs[line, 20] = outputs_half[line, 10]
		outputs[line, 21] = outputs_half[line, 11]
		outputs[line, 22] = (axis-outputs_half[line,10]) + axis - inputs[line, 58]
		outputs[line, 23] = outputs_half[line, 11]

	return outputs
