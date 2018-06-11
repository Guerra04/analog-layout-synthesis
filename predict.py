import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import dataset_processing as proc
import dataset_dependent as dd
import validation as val

N_DEVICES = 12
N_TEMPLATES = 1
TEMPLATE = 0

data = proc.read_file("dataset.csv")
data = dd.to_float(data)
#data_unscaled = dd.normalize(data)

w = dd.get_vector(data, "w")
l = dd.get_vector(data, "l")
nf = dd.get_vector(data, "nf")
wt = dd.get_vector(data, "wt")
ht = dd.get_vector(data, "ht")
x = dd.get_vector(data, "x")
y = dd.get_vector(data, "y")

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

scaler_in = StandardScaler()
scaler_in.fit(inputs)
inputs = scaler_in.transform(inputs)

outputs = []
x_ = np.asarray(x)
y_ = np.asarray(y)
outputs.append(x_[:,TEMPLATE,:])
outputs.append(y_[:,TEMPLATE,:])
n_outputs = 2 * N_DEVICES
outputs_ = np.asarray(outputs)
outputs = outputs_
outputs = outputs.transpose()
#outputs = np.reshape(outputs, (-1, n_outputs))
outputs = proc.reshape(outputs, n_outputs)

scaler_out = StandardScaler()
scaler_out.fit(outputs)
outputs = scaler_out.transform(outputs)

LINE = 1000
feed = inputs[LINE:LINE+1, :]

with tf.Session() as sess:
	file = "./Models/4hidden_template_" + str(TEMPLATE) + ".ckpt"
	file_meta = file + ".meta"

	new_saver = tf.train.import_meta_graph(file_meta)
	new_saver.restore(sess, tf.train.latest_checkpoint('./Models/'))

	#sess.run(tf.global_variables_initializer())

	graph = tf.get_default_graph()

	input = graph.get_tensor_by_name('inputs/input:0')
	regression = graph.get_tensor_by_name('loss/predict:0')

	predict = regression.eval(feed_dict = {input:feed})

	predict = scaler_out.inverse_transform(predict)
	outputs = scaler_out.inverse_transform(outputs)
	feed = scaler_in.inverse_transform(feed)


	#outputs_des = outputs_des/5e-6
	feed = feed/5e-6
	predict = predict/5e-6

	fig,ax = plt.subplots()
	rect_des = []
	rect_pred = []
	rect = []

	#print(outputs[LINE, :]/5e-6)
	for n in range(N_DEVICES):
		print(n, "DES x:", outputs[LINE, 2*n], "y:", outputs[LINE, 1+2*n])
		print(n, "PRED: x:", predict[0, 2*n]*5e-6, "y:", predict[0, 1+2*n]*5e-6)
		rect.append(val.draw_rectangle(outputs[LINE, 2*n]/5e-6, outputs[LINE, 1+2*n]/5e-6, feed[0, 3+5*n], feed[0, 4+5*n], 'r'))
		rect_pred.append(val.draw_rectangle(predict[0, 2*n], predict[0, 1+2*n], feed[0, 3+5*n], feed[0, 4+5*n], 'b'))

	for r in rect:
		ax.add_patch(r)

	for r in rect_pred:
		ax.add_patch(r)

	plt.show()
