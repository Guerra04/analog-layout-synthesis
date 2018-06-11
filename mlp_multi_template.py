import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
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

def get_next_batch(inputs, outputs, batch, batch_size):
	begin = batch*batch_size
	end = begin + batch_size
	x_batch = inputs[begin:end,:]
	y_batch = outputs[begin:end,:]
	return x_batch, y_batch

def main():
	N_DEVICES = 12

	N_HIDDEN = 2

	N_TEMPLATES = 4

	n_epochs = 2000
	batch_size = 50

	degree = 2

	# Read inputs and outputs
	data_train = proc.read_file("DATASETS/dataset_" + str(N_TEMPLATES) + "best_template_train.csv")
	poly = PolynomialFeatures(degree, interaction_only=True)

	inputs_train = dd.get_inputs(data_train)
	#PolynomialFeatures
	inputs_train = poly.fit_transform(inputs_train)
	outputs_train = dd.get_outputs(data_train)

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
	#PolynomialFeatures
	inputs_test = poly.fit_transform(inputs_test)
	outputs_test = dd.get_outputs(data_test)

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

		#outputs
	with tf.name_scope("outputs"):
		#output = tf.placeholder(tf.float32, shape = (None, n_outputs), name = "output")
		output = tf.placeholder(tf.float32, shape = (None, n_outputs), name = "output")
		to_predict = tf.Variable(tf.zeros([1, n_outputs]), validate_shape=False, dtype = tf.float32, name = "to_predict")
		#overlap = tf.placeholder(tf.float32, shape = (None, 1), name = "overlap")


	# NN architecture
		#regression
	hiddens = []
	n_neurons = [50, 100, 250, 500, 750]
	with tf.name_scope("dnn"):
		if(N_HIDDEN == 0):
			regression = fully_connected(input, n_outputs, scope="regression")
		else:
			for n in range(N_HIDDEN):
				name = "hidden" + str(n+1)
				if n == 0:
					hiddens.append(fully_connected(input, n_neurons[n], scope=name))
				else:
					hiddens.append(fully_connected(hiddens[n-1], n_neurons[n], scope=name))
			regression = fully_connected(hiddens[N_HIDDEN-1], n_outputs, scope="regression")

	# Train specification
	learning_rate = 0.01

		#cost function
	with tf.name_scope("loss"):
		to_predict = tf.assign(to_predict, regression, name="predict") #to access in predictor
		#loss = tf.reduce_mean(tf.squared_difference(regression, output), name = "loss")

		error = tf.reduce_mean(tf.squared_difference(regression, output), name = "error")
		#--------------------------------OVERLAP--------------------------------
		xmin = tf.Variable(tf.zeros([N_DEVICES, 1]), dtype=tf.float32)
		xmax = tf.Variable(tf.zeros([N_DEVICES, 1]), dtype=tf.float32)
		ymin = tf.Variable(tf.zeros([N_DEVICES, 1]), dtype=tf.float32)
		ymax = tf.Variable(tf.zeros([N_DEVICES, 1]), dtype=tf.float32)

		indexes_xmin = np.zeros([N_DEVICES, n_outputs], dtype=np.float32)
		for i in range(N_DEVICES):
			indexes_xmin[:, 2*i] = 1
		indexes_ymin = np.zeros([N_DEVICES, n_outputs], dtype=np.float32)
		for i in range(N_DEVICES):
			indexes_ymin[:, 1+2*i] = 1

		indexes_xmax = np.zeros([N_DEVICES, n_inputs], dtype=np.float32)
		for i in range(N_DEVICES):
			indexes_xmax[:, 3+5*i] = 1
		indexes_ymax = np.zeros([N_DEVICES, n_inputs], dtype=np.float32)
		for i in range(N_DEVICES):
			indexes_ymax[:, 4+5*i] = 1

		x_min = tf.matmul(indexes_xmin, tf.transpose(regression))
		y_min = tf.matmul(indexes_ymin, tf.transpose(regression))
		x_max = tf.add(x_min, tf.matmul(indexes_xmax, tf.transpose(input)))
		y_max = tf.add(y_min, tf.matmul(indexes_ymax, tf.transpose(input)))

		xImin = tf.maximum(xmin, tf.transpose(xmin))
		yImin = tf.maximum(ymin, tf.transpose(ymin))

		xImax = tf.minimum(xmax, tf.transpose(xmax))
		yImax = tf.minimum(ymax, tf.transpose(ymax))

		overlap = tf.reduce_sum(tf.multiply(tf.maximum(tf.subtract(xImax, xImin), 0), tf.maximum(tf.subtract(yImax, yImin), 0)))

		alpha = 1.0
		beta = 10.0
		loss = tf.add(tf.multiply(alpha, error), tf.multiply(beta, overlap), name= "loss")
		#-----------------------------------------------------------------------

		#optimizer
	with tf.name_scope("train"):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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
				sess.run(training_op, feed_dict = {input: x_batch, output: y_batch})

			#test_max_error = 0
			#acc_train = sess.run(loss,  feed_dict = {input: x_batch, output: y_batch})
			#acc_test = sess.run(loss,  feed_dict = {input: inputs_test, output: outputs_test})
			if(epoch % 100 == 0):


				acc_train = loss.eval(feed_dict = {input: x_batch, output: y_batch})
				acc_test = loss.eval(feed_dict = {input: inputs_test, output: outputs_test})
				test_max_error = max_error.eval(feed_dict = {input: inputs_test, output: outputs_test})
				print(epoch, "Train error:", acc_train, "Test error:", acc_test, "Test Max Error:", test_max_error)
				save_path = saver.save(sess, "./tmp/my_model.ckpt")
				summary_str = mse_summary.eval(feed_dict = {input: x_batch, output: y_batch})
				step = epoch * n_batches + batch_index
				file_writer.add_summary(summary_str, step)

		file = "./Models/" + str(N_TEMPLATES) + "multi" + str(degree) + "poly" + str(N_HIDDEN) + "hidden_" + str(n_epochs) + ".ckpt"
		save_path = saver.save(sess, file)

	file_writer.close()

if __name__ == '__main__':
	main()
