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

	#--------------METRICS---------------------
	metrics_train = dd.get_metrics(data_train)
	templates_train = np.zeros((3, len(metrics_train)))
	templates_train[0,:] = metrics_train[:, 12]
	templates_train[1,:] = metrics_train[:, 13]
	templates_train[2,:] = metrics_train[:, 14]


	count = np.zeros((3, N_TEMPLATES))
	for i in range(len(metrics_train)):
		for metric in range(3):
			count[metric, int(templates_train[metric, i])] += 1

	count = count/len(metrics_train)*100
	print("-----Training Stats-----")
	print("...Wasted Area")
	for i in range(N_TEMPLATES):
		print(i, ":", count[0, i], "%")
	print("...Maximum AR")
	for i in range(N_TEMPLATES):
		print(i, ":", count[1, i], "%")
	print("...Minimum AR")
	for i in range(N_TEMPLATES):
		print(i, ":", count[2, i], "%")
	print("---------------------")

	metrics_test = dd.get_metrics(data_test)
	templates_test = np.zeros((3, len(metrics_test)))
	templates_test[0,:] = metrics_test[:, 12]
	templates_test[1,:] = metrics_test[:, 13]
	templates_test[2,:] = metrics_test[:, 14]

	count = np.zeros((3, N_TEMPLATES))
	for i in range(len(metrics_test)):
		for metric in range(3):
			count[metric, int(templates_test[metric, i])] += 1

	count = count/len(metrics_test)*100
	print("-----Test Stats-----")
	print("...Wasted Area")
	for i in range(N_TEMPLATES):
		print(i, ":", count[0, i], "%")
	print("...Maximum AR")
	for i in range(N_TEMPLATES):
		print(i, ":", count[1, i], "%")
	print("...Minimum AR")
	for i in range(N_TEMPLATES):
		print(i, ":", count[2, i], "%")
	print("---------------------")
	#----------------------------------------

	with tf.Session() as sess:
		#file = "./Models/" + str(N_TEMPLATES) + "multi" + str(degree) + "poly" + str(N_HIDDEN) + "hidden_" + str(n_epochs) + ".ckpt"
		#file = "./Models/" + str(N_TEMPLATES) + "multi" + str(degree) + "poly" + str(N_HIDDEN) + "hidden" + str(n_epochs) + "beta" + str(beta) + ".ckpt"
		file = "./Models/" + str(N_HIDDEN) + "_" + str(n_epochs) + "_" + str(batch_size) + "_" + str(degree) + "_" + str(keep_prob) + "_" + str(N_TEMPLATES) + ".ckpt"
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

		acc_error_train = np.zeros((3,1))
		acc_error_test = np.zeros((3,1))
		acc_overlap_train = np.zeros((3,1))
		acc_overlap_test = np.zeros((3,1))

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

		error_th = 1.5e-5
		overlap_th = 0

		for line in range(n_train):
			for metric in range(3):
				for n in range(N_DEVICES):
					predict_train_metrics[metric, line, 2*n] = predict_train[line, (2*metric) + (6*n)] #x
					predict_train_metrics[metric, line, 2*n+1] = predict_train[line, 1 + (2*metric) + (6*n)] #y

					outputs_train_metrics[metric, line, 2*n] = outputs_train[line, (2 * metric) + (6*n)] #x
					outputs_train_metrics[metric, line, 2*n+1] = outputs_train[line, 1 + (2*metric) + (6*n)] #y

				error = compute_error(predict_train_metrics[metric, line,:], outputs_train_metrics[metric, line,:])
				error_train[metric, line] = error

				if error_train[metric, line] <= error_th:
					acc_error_train[metric] += 1

				overlap = compute_overlap(predict_train_metrics[metric, line,:], inputs_train[line,1:])
				overlap_train[metric, line] = overlap

				if overlap_train[metric, line] > overlap_th:
					acc_overlap_train[metric] += 1

		for line in range(n_test):
			for metric in range(3):
				for n in range(N_DEVICES):
					predict_test_metrics[metric, line, 2*n] = predict_test[line, (2*metric) + (6*n)] #x
					predict_test_metrics[metric, line, 2*n+1] = predict_test[line, 1 + (2*metric) + (6*n)] #y

					outputs_test_metrics[metric, line, 2*n] = outputs_test[line, (2 * metric) + (6*n)] #x
					outputs_test_metrics[metric, line, 2*n+1] = outputs_test[line, 1 + (2*metric) + (6*n)] #y

				error = compute_error(predict_test_metrics[metric, line,:], outputs_test_metrics[metric, line,:])
				error_test[metric, line] = error

				if error_test[metric, line] <= error_th:
					acc_error_test[metric] += 1

				overlap = compute_overlap(predict_test_metrics[metric, line,:], inputs_test[line,1:])
				overlap_test[metric, line] = overlap

				if overlap_test[metric, line] > overlap_th:
					acc_overlap_test[metric] += 1

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

		print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
		print("||||||||||||||||||||||||||||| TEST ||||||||||||||||||||||||||||")
		print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

		print(".....Wasted Area.....")
		print("Mean Error:", np.mean(error_test[0,:]))
		print("Max Error:", error_test[0, sorted_error_test[0, n_test-1]])
		print("Error Accuracy:", acc_error_test[0]/n_test*100, "%")
		print("Mean Overlap:", np.mean(overlap_test[0,:]))
		print("Max Overlap:", overlap_test[0, sorted_overlap_test[0, n_test-1]])
		print("Overlap Accuracy:", acc_overlap_test[0]/n_test*100, "%")
		print(".....Minimum AR.....")
		print("Mean Error:", np.mean(error_test[1,:]))
		print("Max Error:", error_test[1, sorted_error_test[1, n_test-1]])
		print("Error Accuracy:", acc_error_test[1]/n_test*100, "%")
		print("Mean Overlap:", np.mean(overlap_test[1,:]))
		print("Max Overlap:", overlap_test[1, sorted_overlap_test[1, n_test-1]])
		print("Overlap Accuracy:", acc_overlap_test[1]/n_test*100, "%")
		print(".....Maximum AR.....")
		print("Mean Error:", np.mean(error_test[2,:]))
		print("Max Error:", error_test[2, sorted_error_test[2, n_test-1]])
		print("Error Accuracy:", acc_error_test[2]/n_test*100, "%")
		print("Mean Overlap:", np.mean(overlap_test[2,:]))
		print("Max Overlap:", overlap_test[2, sorted_overlap_test[2, n_test-1]])
		print("Overlap Accuracy:", acc_overlap_test[2]/n_test*100, "%")

		print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
		print("|||||||||||||||||||||||||||| TRAIN ||||||||||||||||||||||||||||")
		print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
		print(".....Wasted Area.....")
		print("Mean Error:", np.mean(error_train[0,:]))
		print("Max Error:", error_train[0, sorted_error_train[0, n_train-1]])
		print("Error Accuracy:", acc_error_train[0]/n_train*100, "%")
		print("Mean Overlap:", np.mean(overlap_train[0,:]))
		print("Max Overlap:", overlap_train[0, sorted_overlap_train[0, n_train-1]])
		print("Overlap Accuracy:", acc_overlap_train[0]/n_train*100, "%")
		print(".....Minimum AR.....")
		print("Mean Error:", np.mean(error_train[1,:]))
		print("Max Error:", error_train[1, sorted_error_train[1, n_train-1]])
		print("Error Accuracy:", acc_error_train[1]/n_train*100, "%")
		print("Mean Overlap:", np.mean(overlap_train[1,:]))
		print("Max Overlap:", overlap_train[1, sorted_overlap_train[1, n_train-1]])
		print("Overlap Accuracy:", acc_overlap_train[1]/n_train*100, "%")
		print(".....Maximum AR.....")
		print("Mean Error:", np.mean(error_train[2,:]))
		print("Max Error:", error_train[2, sorted_error_train[2, n_train-1]])
		print("Error Accuracy:", acc_error_train[2]/n_train*100, "%")
		print("Mean Overlap:", np.mean(overlap_train[2,:]))
		print("Max Overlap:", overlap_train[2, sorted_overlap_train[2, n_train-1]])
		print("Overlap Accuracy:", acc_overlap_train[2]/n_train*100, "%")

		'''
		to_check = 1
		#----------------------------TRAIN------------------------------
		for i in range(to_check):
			for metric in range(3):
				#idx = sorted_error_train[metric, n_train-i-1] #worst
				idx = sorted_error_train[metric, math.floor(3*n_train/4)] #middle
				#idx = sorted_error_train[metric, i] #better
				#idx = sorted_overlap_train[metric, n_train-i-1] #max overlap

				print("Template:", int(templates_train[metric, idx]))
				fig,ax = plt.subplots()
				title = "Train: " + str(i)
				fig.suptitle(title, fontsize=16)
				rect_pred = []
				rect = []

				for n in range(N_DEVICES):
					rect.append(val.draw_rectangle(outputs_train_metrics[metric, idx, 2*n], outputs_train_metrics[metric, idx, 1+2*n], inputs_train[idx, 3+5*n], inputs_train[idx, 4+5*n], 'r'))
					rect_pred.append(val.draw_rectangle(predict_train_metrics[metric, idx, 2*n], predict_train_metrics[metric, idx, 1+2*n], inputs_train[idx, 3+5*n], inputs_train[idx, 4+5*n], 'b'))

				for r in rect:
					ax.add_patch(r)

				for r in rect_pred:
					ax.add_patch(r)

				plt.show()

		#------------------------------TEST---------------------
		for i in range(to_check):
			for metric in range(3):
				#idx = sorted_error_test[metric, n_test-i-1] #worst
				idx = sorted_error_test[metric, math.floor(3*n_test/4)] #middle
				#idx = sorted_error_test[metric, i] #better
				#idx = sorted_overlap_test[metric, n_test-i-1] #max overlap

				print("Template:", int(templates_test[metric, idx]))
				fig,ax = plt.subplots()
				title = "Test: " + str(i)
				fig.suptitle(title, fontsize=16)
				rect_pred = []
				rect = []

				for n in range(N_DEVICES):
					rect.append(val.draw_rectangle(outputs_test_metrics[metric, idx, 2*n], outputs_test_metrics[metric, idx, 1+2*n], inputs_test[idx, 3+5*n], inputs_test[idx, 4+5*n], 'r'))
					rect_pred.append(val.draw_rectangle(predict_test_metrics[metric, idx, 2*n], predict_test_metrics[metric, idx, 1+2*n], inputs_test[idx, 3+5*n], inputs_test[idx, 4+5*n], 'b'))

				for r in rect:
					ax.add_patch(r)

				for r in rect_pred:
					ax.add_patch(r)

				plt.show()
		'''

if __name__ == '__main__':
	main()
