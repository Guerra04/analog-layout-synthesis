import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import dataset_processing as proc
import dataset_dependent_multi as dd
import validation as val
from Rectangle import compute_overlap
from Scaler import Scaler

from HYPERPARAMS import *

'''
N_DEVICES = 12
N_TEMPLATES = 4
N_HIDDEN = 4

epochs = 2000
degree = 2

n_epochs = 2000;
'''

def compute_error(predict, target):
	error = 0
	for i in range(N_DEVICES):
		aux = np.sqrt((predict[2*i] - target[2*i])**2 + (predict[1+2*i] - target[1+2*i])**2)
		error = error + aux

	return error

def main():
	# Read inputs and outputs
	data_train = proc.read_file("DATASETS/dataset_" + str(N_TEMPLATES) + "best_template_train.csv")
	poly = PolynomialFeatures(degree, interaction_only=True)

	inputs_train = dd.get_inputs(data_train)
	outputs_train = dd.get_outputs(data_train)
	#outputs_train = outputs_train + 1e-20*np.random.normal(size=outputs_train.shape)
	#center data
	outputs_train = dd.center_data(inputs_train, outputs_train)

	#PolynomialFeatures
	inputs_train = poly.fit_transform(inputs_train)

	#scaler_in = StandardScaler()
	#scaler_in.fit(inputs_train)
	#scaler_in = MinMaxScaler()
	#scaler_in.fit(inputs_train)
	scaler_in = Scaler(inputs_train)
	inputs_train = scaler_in.transform(inputs_train)

	#scaler_out = StandardScaler()
	#scaler_out.fit(outputs_train)
	#scaler_out = MinMaxScaler()
	#scaler_out.fit(outputs_train)
	scaler_out = Scaler(outputs_train)
	outputs_train = scaler_out.transform(outputs_train)

	data_test = proc.read_file("DATASETS/dataset_" + str(N_TEMPLATES) + "best_template_test.csv")

	inputs_test = dd.get_inputs(data_test)
	outputs_test = dd.get_outputs(data_test)
	#center data
	outputs_test = dd.center_data(inputs_test, outputs_test)

	#PolynomialFeatures
	inputs_test = poly.fit_transform(inputs_test)

	inputs_test = scaler_in.transform(inputs_test)
	outputs_test = scaler_out.transform(outputs_test)

	n_train = len(inputs_train) #number of examples
	n_test = len(inputs_test)
	n_inputs = inputs_train.shape[1]
	n_outputs = outputs_train.shape[1]

	#--------------METRICS---------------------
	metrics_train = dd.get_metrics(data_train)
	count = np.zeros((N_TEMPLATES, 1))
	for i in range(len(metrics_train)):
		count[int(metrics_train[i, 4])] += 1

	count = count/len(metrics_train)*100
	print("Training Stats:")
	for i in range(N_TEMPLATES):
		print(i, ":", count[i], "%")
	print("---------------------")

	metrics_test = dd.get_metrics(data_test)
	count = np.zeros((N_TEMPLATES, 1))
	for i in range(len(metrics_test)):
		count[int(metrics_test[i, 4])] += 1

	count = count/len(metrics_test)*100
	print("Test Stats:")
	for i in range(N_TEMPLATES):
		print(i, ":", count[i], "%")
	print("---------------------")
	#----------------------------------------

	with tf.Session() as sess:
		#file = "./Models/" + str(N_TEMPLATES) + "multi" + str(degree) + "poly" + str(N_HIDDEN) + "hidden_" + str(n_epochs) + ".ckpt"
		file = "./Models/" + str(N_TEMPLATES) + "multi" + str(degree) + "poly" + str(N_HIDDEN) + "hidden" + str(n_epochs) + "beta" + str(beta) + ".ckpt"
		file_meta = file + ".meta"

		new_saver = tf.train.import_meta_graph(file_meta)
		new_saver.restore(sess, tf.train.latest_checkpoint('./Models/'))

		graph = tf.get_default_graph()

		input = graph.get_tensor_by_name('inputs/input:0')
		regression = graph.get_tensor_by_name('loss/predict:0')
		output = graph.get_tensor_by_name('outputs/output:0')
		loss = graph.get_tensor_by_name('loss/loss:0')

		predict_train = np.zeros((n_train, n_outputs))
		predict_test = np.zeros((n_test, n_outputs))
		error_train = np.zeros(n_train)
		error_test = np.zeros(n_test)
		overlap_train = np.zeros(n_train)
		overlap_test = np.zeros(n_test)

		for line in range(n_train):
			feed = inputs_train[line:line+1, :]

			predict = regression.eval(feed_dict = {input:feed})
			predict_train[line, :] = predict

		for line in range(n_test):
			feed = inputs_test[line:line+1, :]

			predict = regression.eval(feed_dict = {input:feed})
			predict_test[line, :] = predict

		predict_train = scaler_out.inverse_transform(predict_train)
		predict_test = scaler_out.inverse_transform(predict_test)

		inputs_train = scaler_in.inverse_transform(inputs_train)
		inputs_test = scaler_in.inverse_transform(inputs_test)

		outputs_train = scaler_out.inverse_transform(outputs_train)
		outputs_test = scaler_out.inverse_transform(outputs_test)

		for line in range(n_train):
			error = compute_error(predict_train[line,:], outputs_train[line,:])
			error_train[line] = error
			overlap = compute_overlap(predict_train[line,:], inputs_train[line,1:])
			overlap_train[line] = overlap

		for line in range(n_test):
			error = compute_error(predict_test[line,:], outputs_test[line,:])
			error_test[line] = error
			overlap = compute_overlap(predict_test[line,:], inputs_test[line,1:])
			overlap_test[line] = overlap

		#Normalize for easier visualization
		predict_train = predict_train/5e-6
		predict_test = predict_test/5e-6

		inputs_train = inputs_train[:,1:]/5e-6
		inputs_test = inputs_test[:,1:]/5e-6

		outputs_train = outputs_train/5e-6
		outputs_test = outputs_test/5e-6

		sorted_error_train = np.argsort(error_train)
		sorted_error_test = np.argsort(error_test)
		sorted_overlap_train = np.argsort(overlap_train)
		sorted_overlap_test = np.argsort(overlap_test)

		print("-----------------------------TRAIN-----------------------------")
		print("Mean Error:", np.mean(error_train))
		print("Max Error:", error_train[sorted_error_train[n_train-1]])
		print("Mean Overlap:", np.mean(overlap_train))
		print("Max Overlap:", overlap_train[sorted_overlap_train[n_train-1]])
		print("------------------------------TEST-----------------------------")
		print("Mean Error:", np.mean(error_test))
		print("Max Error:", error_test[sorted_error_test[n_test-1]])
		print("Mean Overlap:", np.mean(overlap_test))
		print("Max Overlap:", overlap_test[sorted_overlap_test[n_test-1]])
		print("---------------------------------------------------------------")

		to_check = 1
		#----------------------------TRAIN------------------------------
		for i in range(to_check):
			idx = sorted_error_train[n_train-i-1] #worst
			#idx = sorted_error_train[math.floor(n_train/2)] #middle
			#idx = sorted_error_train[i] #better
			#idx = sorted_overlap_train[n_train-i-1] #max overlap

			print("Template:", int(metrics_train[idx, 4]))
			fig,ax = plt.subplots()
			title = "Train: " + str(i)
			fig.suptitle(title, fontsize=16)
			rect_pred = []
			rect = []

			for n in range(N_DEVICES):
				rect.append(val.draw_rectangle(outputs_train[idx, 2*n], outputs_train[idx, 1+2*n], inputs_train[idx, 3+5*n], inputs_train[idx, 4+5*n], 'r'))
				rect_pred.append(val.draw_rectangle(predict_train[idx, 2*n], predict_train[idx, 1+2*n], inputs_train[idx, 3+5*n], inputs_train[idx, 4+5*n], 'b'))
				'''
				print("TARGET ||| PREDICT")
				print("-----", n, "-----")
				print("x", outputs_train[idx, 2*n], "|||", predict_train[idx, 2*n])
				print("y", outputs_train[idx, 1+2*n], "|||", predict_train[idx, 1+2*n])
				'''

			for r in rect:
				ax.add_patch(r)

			for r in rect_pred:
				ax.add_patch(r)

			plt.show()

		#------------------------------TEST---------------------
		for i in range(to_check):
			idx = sorted_error_test[n_test-i-1] #worst
			#idx = sorted_error_test[math.floor(n_test/2)] #middle
			#idx = sorted_error_test[i] #better
			#idx = sorted_overlap_test[n_test-i-1] #max overlap

			print("Template:", int(metrics_test[idx, 4]))
			fig,ax = plt.subplots()
			title = "Test: " + str(i)
			fig.suptitle(title, fontsize=16)
			rect_pred = []
			rect = []

			for n in range(N_DEVICES):
				rect.append(val.draw_rectangle(outputs_test[idx, 2*n], outputs_test[idx, 1+2*n], inputs_test[idx, 3+5*n], inputs_test[idx, 4+5*n], 'r'))
				rect_pred.append(val.draw_rectangle(predict_test[idx, 2*n], predict_test[idx, 1+2*n], inputs_test[idx, 3+5*n], inputs_test[idx, 4+5*n], 'b'))
				'''
				print("TARGET ||| PREDICT")
				print("-----", n, "-----")
				print("x", outputs_test[idx, 2*n], "|||", predict_test[idx, 2*n])
				print("y", outputs_test[idx, 1+2*n], "|||", predict_test[idx, 1+2*n])
				'''

			for r in rect:
				ax.add_patch(r)

			for r in rect_pred:
				ax.add_patch(r)

			plt.show()


if __name__ == '__main__':
	main()
