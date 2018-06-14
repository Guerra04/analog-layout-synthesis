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

def compute_error(predict, target):
	error = 0
	for i in range(N_DEVICES):
		aux = np.sqrt((predict[0+2*i] - target[0+2*i])**2 + (predict[1+2*i] - target[1+2*i])**2)

		error = error + aux

	return error

def main():
	# Read inputs and outputs
	data_train = proc.read_file("DATASETS/dataset_" + str(N_TEMPLATES) + "best_template_metrics_train.csv")
	poly = PolynomialFeatures(degree, interaction_only=True)

	inputs_train = dd.get_inputs(data_train)
	outputs_train = dd.get_outputs(data_train)

	#center data
	outputs_train = dd.center_data(inputs_train, outputs_train)

	#PolynomialFeatures
	inputs_train = poly.fit_transform(inputs_train)

	scaler_in = Scaler(inputs_train)
	inputs_train = scaler_in.transform(inputs_train)

	scaler_out = Scaler(outputs_train)
	outputs_train = scaler_out.transform(outputs_train)

	data_test = proc.read_file("DATASETS/dataset_" + str(N_TEMPLATES) + "best_template_metrics_test.csv")

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
	n_outputs_metric = int(n_outputs/3)

	with tf.Session() as sess:
		#file = "./Models/" + str(N_TEMPLATES) + "multi" + str(degree) + "poly" + str(N_HIDDEN) + "hidden_" + str(n_epochs) + ".ckpt"
		file = "./Models/" + str(N_TEMPLATES) + "multi" + str(degree) + "poly" + str(N_HIDDEN) + "hidden" + str(n_epochs) + "beta" + str(beta) + ".ckpt"
		file_meta = file + ".meta"

		new_saver = tf.train.import_meta_graph(file_meta)
		new_saver.restore(sess, tf.train.latest_checkpoint('./Models/'))

		graph = tf.get_default_graph()

		input = graph.get_tensor_by_name('inputs/input:0')
		is_training = graph.get_tensor_by_name('inputs/is_training:0')
		regression = graph.get_tensor_by_name('loss/predict:0')
		output = graph.get_tensor_by_name('outputs/output:0')
		loss = graph.get_tensor_by_name('loss/loss:0')

		predict_train = np.zeros((n_train, n_outputs))
		predict_test = np.zeros((n_test, n_outputs))
		predict_train_metrics = np.zeros((3, n_train, n_outputs_metric))
		predict_test_metrics = np.zeros((3, n_test, n_outputs_metric))
		outputs_train_metrics = np.zeros((3, n_train, n_outputs_metric))
		outputs_test_metrics = np.zeros((3, n_test, n_outputs_metric))
		error_train = np.zeros((3, n_train))
		error_test = np.zeros((3, n_test))
		overlap_train = np.zeros((3, n_train))
		overlap_test = np.zeros((3, n_test))

		for line in range(n_train):
			feed = inputs_train[line:line+1, :]

			predict = regression.eval(feed_dict = {input:feed, is_training: False})
			predict_train[line, :] = predict

		for line in range(n_test):
			feed = inputs_test[line:line+1, :]

			predict = regression.eval(feed_dict = {input:feed, is_training: False})
			predict_test[line, :] = predict

		predict_train = scaler_out.inverse_transform(predict_train)
		predict_test = scaler_out.inverse_transform(predict_test)

		inputs_train = scaler_in.inverse_transform(inputs_train)
		inputs_test = scaler_in.inverse_transform(inputs_test)

		outputs_train = scaler_out.inverse_transform(outputs_train)
		outputs_test = scaler_out.inverse_transform(outputs_test)

		for line in range(n_train):
			for metric in range(3):
				for n in range(N_DEVICES):
					predict_train_metrics[metric, line, 2*n] = predict_train[line, (2*metric) + (6*n)] #x
					predict_train_metrics[metric, line, 2*n+1] = predict_train[line, 1 + (2*metric) + (6*n)] #y

					outputs_train_metrics[metric, line, 2*n] = outputs_train[line, (2 * metric) + (6*n)] #x
					outputs_train_metrics[metric, line, 2*n+1] = outputs_train[line, 1 + (2*metric) + (6*n)] #y

				error = compute_error(predict_train_metrics[metric, line,:], outputs_train_metrics[metric, line,:])
				error_train[metric, line] = error

				overlap = compute_overlap(predict_train_metrics[metric, line,:], inputs_train[line,1:])
				overlap_train[metric, line] = overlap

		for line in range(n_test):
			for metric in range(3):
				for n in range(N_DEVICES):
					predict_test_metrics[metric, line, 2*n] = predict_test[line, (2*metric) + (6*n)] #x
					predict_test_metrics[metric, line, 2*n+1] = predict_test[line, 1 + (2*metric) + (6*n)] #y

					outputs_test_metrics[metric, line, 2*n] = outputs_test[line, (2 * metric) + (6*n)] #x
					outputs_test_metrics[metric, line, 2*n+1] = outputs_test[line, 1 + (2*metric) + (6*n)] #y

				error = compute_error(predict_test_metrics[metric, line,:], outputs_test_metrics[metric, line,:])
				error_test[metric, line] = error

				overlap = compute_overlap(predict_test_metrics[metric, line,:], inputs_test[line,1:])
				overlap_test[metric, line] = overlap

		#Normalize for easier visualization
		predict_train_metrics = predict_train_metrics/5e-6
		predict_test_metrics = predict_test_metrics/5e-6

		inputs_train = inputs_train[:,1:]/5e-6
		inputs_test = inputs_test[:,1:]/5e-6

		outputs_train_metrics = outputs_train_metrics/5e-6
		outputs_test_metrics = outputs_test_metrics/5e-6

		sorted_error_train = np.argsort(error_train)
		sorted_error_test = np.argsort(error_test)
		sorted_overlap_train = np.argsort(overlap_train)
		sorted_overlap_test = np.argsort(overlap_test)

		metric = 0
		for i in range(n_train):
			idx = sorted_error_train[metric, 8000-i-1] #middle

			print("Error:", error_train[metric, idx])
			fig,ax = plt.subplots()
			title = "Train: " + str(i)
			fig.suptitle(title, fontsize=16)
			rect_pred = []
			rect = []

			for n in range(N_DEVICES):
				rect.append(val.draw_rectangle(outputs_train_metrics[metric, idx, 2*n], outputs_train_metrics[metric, idx, 1+2*n], inputs_train[idx, 3+5*n], inputs_train[idx, 4+5*n], 'r'))
				rect_pred.append(val.draw_rectangle(predict_train_metrics[metric, idx, 2*n], predict_train_metrics[metric, idx, 1+2*n], inputs_train[idx, 3+5*n], inputs_train[idx, 4+5*n], 'b'))
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

if __name__ == '__main__':
	main()


#error_th = 1.5e-5
