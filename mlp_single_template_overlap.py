import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
import Rectangle as rect

from datetime import datetime

import dataset_processing as proc
import dataset_dependent as dd

def get_next_batch(inputs, outputs, batch, batch_size):
	begin = batch*batch_size
	end = begin + batch_size
	x_batch = inputs[begin:end,:]
	y_batch = outputs[begin:end,:]
	return x_batch, y_batch

def main():
	N_DEVICES = 12
	N_TEMPLATES = 1

	TEMPLATE = 0
	N_HIDDEN = 3

	n_epochs = 1000
	batch_size = 50

	# Read inputs and outputs
	data_train = proc.read_file("dataset_train.csv")

	inputs_train = dd.get_inputs(data_train)
	outputs_train = dd.get_outputs(data_train, TEMPLATE)

	scaler_in = StandardScaler()
	scaler_in.fit(inputs_train)
	inputs_train = scaler_in.transform(inputs_train)

	scaler_out = StandardScaler()
	scaler_out.fit(outputs_train)
	outputs_train = scaler_out.transform(outputs_train)

	data_test = proc.read_file("dataset_test.csv")

	inputs_test = dd.get_inputs(data_test)
	outputs_test = dd.get_outputs(data_test, TEMPLATE)

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
	n_neurons = [50, 100, 250, 500]
	with tf.name_scope("dnn"):
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
		mean_error = tf.reduce_mean(tf.squared_difference(regression, output), name = "mean_error")

		#overlap = tf.py_func(rect.overlap_array, [regression, input], [tf.float32])

		#----------------OVERLAP------------------------------------------------
		overlaps = tf.constant(0.000)
		for i in range(N_DEVICES):
			xmin_a = regression[:, 2*i]
			ymin_a = regression[:, 1+2*i]
			xmax_a = tf.add(xmin_a, input[:, 3+5*i])
			ymax_a = tf.add(ymin_a, input[:, 4+5*i])
			for j in range(i+1, N_DEVICES):
				xmin_b = regression[:, 2*j]
				ymin_b = regression[:, 1+2*j]
				xmax_b = tf.add(xmin_b, input[:, 3+5*j])
				ymax_b = tf.add(ymin_b, input[:, 4+5*j])

				dx = tf.subtract(tf.minimum(xmax_a, xmax_b), tf.maximum(xmin_a, xmin_b))
				dy = tf.subtract(tf.minimum(ymax_a, ymax_b), tf.maximum(ymin_a, ymin_b))
				area = tf.multiply(dx, dy)

				aux1 = tf.greater_equal(dx, tf.zeros_like(dx))
				aux2 = tf.greater_equal(dy, tf.zeros_like(dy))
				cond = tf.logical_and(aux1, aux2)
				overlap = tf.where(cond, area, tf.zeros_like(area))
				overlaps = tf.add(overlaps, overlap)
		#-----------------------------------------------------------------------
		overlap_sum = tf.reduce_sum(overlaps,  name = "overlap_sum")
		#overlap_sum = tf.zeros_like(mean_error)
		#loss = tf.reduce_sum(overlap,  name = "loss")
		w_overlap = 0.01
		w_error = 1.0
		loss = tf.add(tf.multiply(w_error, mean_error), tf.multiply(w_overlap, overlap_sum), name="loss")

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
				'''
				area_eval = area.eval(feed_dict = {input: x_batch, output: y_batch})
				cond_eval = cond.eval(feed_dict = {input: x_batch, output: y_batch})
				over_eval = overlap.eval(feed_dict = {input: x_batch, output: y_batch})
				sum_eval = overlap_sum.eval(feed_dict = {input: x_batch, output: y_batch})
				overlaps_eval = overlaps.eval(feed_dict = {input: x_batch, output: y_batch})
				print("OVERLAPS", overlaps_eval)
				print("SUM", sum_eval)
				'''

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

		file = "./Models/" + str(N_HIDDEN) + "hidden_template_" + str(TEMPLATE) + "_overlap.ckpt"
		save_path = saver.save(sess, file)

	file_writer.close()

if __name__ == '__main__':
	main()
