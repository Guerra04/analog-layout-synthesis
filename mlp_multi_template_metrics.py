import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import dropout
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import Rectangle as rect

from datetime import datetime

import dataset_processing as proc
import dataset_dependent_multi as dd
from Scaler import Scaler

from HYPERPARAMS import *

def get_next_batch(inputs, outputs, batch, batch_size):
	begin = batch*batch_size
	end = begin + batch_size
	x_batch = inputs[begin:end,:]
	y_batch = outputs[begin:end,:]
	return x_batch, y_batch

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

	num_examples = len(inputs_train) #number of examples
	n_inputs = inputs_train.shape[1]
	n_outputs = outputs_train.shape[1]

	# NN variables
		#inputs
	with tf.name_scope("inputs"):
		#input = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "input")
		input = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "input")
		is_training = tf.placeholder(tf.bool, shape=(), name = "is_training")

		#outputs
	with tf.name_scope("outputs"):
		#output = tf.placeholder(tf.float32, shape = (None, n_outputs), name = "output")
		output = tf.placeholder(tf.float32, shape = (None, n_outputs), name = "output")
		to_predict = tf.Variable(tf.zeros([1, n_outputs]), validate_shape=False, dtype = tf.float32, name = "to_predict")
		#overlap = tf.placeholder(tf.float32, shape = (None, 1), name = "overlap")


	# NN architecture
		#regression
	hiddens = []
	hiddens_drop = []
	n_neurons = [100, 200, 400, 600, 800]
	with tf.name_scope("dnn"):
		if(N_HIDDEN == 0):
			regression = fully_connected(input, n_outputs, scope="regression")
		else:
			for n in range(N_HIDDEN):
				input_drop = dropout(input, keep_prob, is_training=is_training)
				name = "hidden" + str(n+1)
				if n == 0:
					hiddens.append(fully_connected(input_drop, n_neurons[n], scope=name, activation_fn=tf.tanh))
				else:
					hiddens.append(fully_connected(hiddens_drop[n-1], n_neurons[n], scope=name, activation_fn=tf.tanh))
				hiddens_drop.append(dropout(hiddens[n], keep_prob, is_training=is_training))

			regression = fully_connected(hiddens_drop[N_HIDDEN-1], n_outputs, scope="regression", activation_fn=None)

	# Train specification
	learning_rate = 0.01

		#cost function
	with tf.name_scope("loss"):
		to_predict = tf.assign(to_predict, regression, name="predict") #to access in predictor
		loss = tf.reduce_mean(tf.squared_difference(regression, output), name = "loss")

		error = tf.reduce_mean(tf.squared_difference(regression, output), name = "error")
		overlap = tf.constant(0)
		'''
		#-----------------------------OVERLAP - STACK OVERFLOW------------------
		indexes_xmin = np.zeros([n_outputs, N_DEVICES], dtype=np.float32)
		for i in range(N_DEVICES):
			indexes_xmin[2*i, i] = 1
		indexes_ymin = np.zeros([n_outputs, N_DEVICES], dtype=np.float32)
		for i in range(N_DEVICES):
			indexes_ymin[1+2*i, i] = 1

		indexes_xmax = np.zeros([n_inputs, N_DEVICES], dtype=np.float32)
		for i in range(N_DEVICES):
			indexes_xmax[3+5*i, i] = 1
		indexes_ymax = np.zeros([n_inputs, N_DEVICES], dtype=np.float32)
		for i in range(N_DEVICES):
			indexes_ymax[4+5*i, i] = 1

		xmin = tf.matmul(regression, indexes_xmin)
		ymin = tf.matmul(regression, indexes_ymin)
		xmax = tf.add(xmin, tf.matmul(input, indexes_xmax))
		ymax = tf.add(ymin, tf.matmul(input, indexes_ymax))

		xmin1 = xmin[:, None] #size: batch_size*1*N_DEVICES
		xmin2 = xmin[:,:,None] #size: batch_size*N_DEVICES*1
		ymin1 = ymin[:, None]
		ymin2 = ymin[:,:,None]
		xmax1 = xmax[:, None]
		xmax2 = xmax[:,:,None]
		ymax1 = ymax[:, None]
		ymax2 = ymax[:,:,None]

		dx = tf.minimum(xmax1, xmax2) - tf.maximum(xmin1, xmin2)
		dy = tf.minimum(ymax1, ymax2) - tf.maximum(ymin1, ymin2)
		areas = tf.multiply(tf.maximum(dx, 0.0), tf.maximum(dy, 0.0))
		mean_areas = tf.reduce_mean(areas, 0)

		overlap = tf.reduce_sum(mean_areas) - tf.reduce_sum(tf.trace(mean_areas))

		#loss = tf.divide(tf.reduce_mean(overlaps), 2.0, name="loss") #só overlap
		loss = tf.add(tf.multiply(alpha, error), tf.multiply(beta, overlap), name= "loss")
		#-----------------------------------------------------------------------
		'''

		#optimizer
	with tf.name_scope("train"):
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		optimizer = tf.train.AdamOptimizer()
		training_op = optimizer.minimize(loss)

		#evaluate
	with tf.name_scope("eval"):
		total_error = loss
		max_error = tf.reduce_max(tf.abs(tf.subtract(regression, output)), name="max")

		#initializer
	init = tf.global_variables_initializer()

		#saver
	saver = tf.train.Saver()

	#log directory
	now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
	root_logdir = "tf_logs"
	logdir = "{}/run-{}/".format(root_logdir, now)

	#build logs
	mse_summary = tf.summary.scalar('MSE', loss)
	file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

	# Train
	n_batches = num_examples // batch_size
	with tf.Session() as sess:
		init.run()
		print("---------------------STARTING TRAIN-------------------------")
		for epoch in range(n_epochs):
			for batch_index in range(n_batches):
				x_batch, y_batch = get_next_batch(inputs_train, outputs_train, batch_index, batch_size)
				sess.run(training_op, feed_dict = {input: x_batch, output: y_batch, is_training: True})

			#test_max_error = 0
			#acc_train = sess.run(loss,  feed_dict = {input: x_batch, output: y_batch})
			#acc_test = sess.run(loss,  feed_dict = {input: inputs_test, output: outputs_test})
			if(epoch % 100 == 0):
				acc_train = loss.eval(feed_dict = {input: x_batch, output: y_batch, is_training: True})
				ol_train = overlap.eval(feed_dict = {input: x_batch, output: y_batch, is_training: True})
				error_train = error.eval(feed_dict = {input: x_batch, output: y_batch, is_training: True})

				acc_test = loss.eval(feed_dict = {input: inputs_test, output: outputs_test, is_training: True})
				ol_test = overlap.eval(feed_dict = {input: inputs_test, output: outputs_test, is_training: True})
				error_test = error.eval(feed_dict = {input: inputs_test, output: outputs_test, is_training: True})

				#debug = mean_areas.eval(feed_dict = {input: x_batch, output: y_batch})
				#print(debug.shape)
				#print(debug)

				print(epoch, "Train loss:", acc_train, "Train error:", error_train, "Train Overlap:", ol_train)
				print(epoch, "Test loss:", acc_test, "Test error:", error_test, "Test Overlap:", ol_test)

				save_path = saver.save(sess, "./tmp/my_model.ckpt")
				summary_str = mse_summary.eval(feed_dict = {input: x_batch, output: y_batch, is_training: True})
				step = epoch * n_batches + batch_index
				file_writer.add_summary(summary_str, step)

		file = "./Models/" + str(N_TEMPLATES) + "multi" + str(degree) + "poly" + str(N_HIDDEN) + "hidden" + str(n_epochs) + "beta" + str(beta) + ".ckpt"
		save_path = saver.save(sess, file)

	file_writer.close()

if __name__ == '__main__':
	main()
