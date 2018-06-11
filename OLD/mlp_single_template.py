import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

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
	TEMPLATE = 2

	# Read inputs and outputs
	data = proc.read_file("dataset.csv")
	data = dd.to_float(data)

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
	outputs = np.reshape(outputs, (-1, n_outputs))

	scaler_out = StandardScaler()
	scaler_out.fit(outputs)
	outputs = scaler_out.transform(outputs)

	#split train and test
	inputs_train, inputs_test, outputs_train, outputs_test = cross_validation.train_test_split(
		inputs, outputs, test_size = 0.2, random_state=0)
	num_examples = len(inputs_train) #number of examples

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


	# NN architecture
		#regression
	N_HIDDEN = 3
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
		to_predict = tf.assign(to_predict, regression, name="predict")
		loss = tf.reduce_mean(tf.squared_difference(regression, output), name = "loss")

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
	n_epochs = 2500
	batch_size = 50
	n_batches = num_examples // batch_size
	with tf.Session() as sess:
		init.run()
		for epoch in range(n_epochs):
			for batch_index in range(n_batches):
				x_batch, y_batch = get_next_batch(inputs_train, outputs_train, batch_index, batch_size)
				sess.run(training_op, feed_dict = {input: x_batch, output: y_batch})


			acc_train = loss.eval(feed_dict = {input: x_batch, output: y_batch})
			acc_test = loss.eval(feed_dict = {input: inputs_test, output: outputs_test})
			test_max_error = max_error.eval(feed_dict = {input: inputs_test, output: outputs_test})
			#test_max_error = 0
			#acc_train = sess.run(loss,  feed_dict = {input: x_batch, output: y_batch})
			#acc_test = sess.run(loss,  feed_dict = {input: inputs_test, output: outputs_test})
			if(epoch % 100 == 0):
				print(epoch, "Train error:", acc_train, "Test error:", acc_test, "Test Max Error:", test_max_error)
				save_path = saver.save(sess, "./tmp/my_model.ckpt")
				summary_str = mse_summary.eval(feed_dict = {input: x_batch, output: y_batch})
				step = epoch * n_batches + batch_index
				file_writer.add_summary(summary_str, step)

		file = "./Models/" + str(N_HIDDEN) + "hidden_template_" + str(TEMPLATE) + ".ckpt"
		save_path = saver.save(sess, file)

	file_writer.close()

if __name__ == '__main__':
	main()
