import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import dropout
from sklearn.preprocessing import PolynomialFeatures
import time
from datetime import datetime
import numpy as np

from dataset_processing import read_file
import dataset_dependent_multi as dd
from HYPERPARAMS import *
from Scaler import Scaler


def get_next_batch(inputs, outputs, batch, batch_size):
	begin = batch*batch_size
	end = begin + batch_size
	x_batch = inputs[begin:end,:]
	y_batch = outputs[begin:end,:]
	return x_batch, y_batch


def apply_dropout(input, keep_prob, is_training):
	if DROPOUT:
		return dropout(input, keep_prob, is_training=is_training)
	else:
		return input


def main():
	# Read inputs and outputs
	data_train = read_file("../DATASETS/dataset_" + str(N_TEMPLATES) + "best_template_metrics_train.csv")
	inputs_train = dd.get_inputs(data_train)
	outputs_train = dd.get_outputs(data_train)

	data_test = read_file("../DATASETS/dataset_" + str(N_TEMPLATES) + "best_template_metrics_test.csv")
	inputs_test = dd.get_inputs(data_test)
	outputs_test = dd.get_outputs(data_test)

	if CENTER:
		outputs_train = dd.center_data(inputs_train, outputs_train)
		outputs_test = dd.center_data(inputs_test, outputs_test)

	# Use only best area placement
	if not METRICS:
		outputs_train_aux = []
		outputs_test_aux = []
		for line in range(len(outputs_train)):
			outputs_train_aux.append([])
			for n in range(N_DEVICES):
				outputs_train_aux[line].append(outputs_train[line, (6*n)]) #x
				outputs_train_aux[line].append(outputs_train[line, 1 + (6*n)]) #y
		for line in range(len(outputs_test)):
			outputs_test_aux.append([])
			for n in range(N_DEVICES):
				outputs_test_aux[line].append(outputs_test[line, (6*n)]) #x
				outputs_test_aux[line].append(outputs_test[line, 1 + (6*n)]) #y

		outputs_train = np.asarray(outputs_train_aux)
		outputs_test = np.asarray(outputs_test_aux)

	if POLY:
		poly = PolynomialFeatures(DEGREE, interaction_only=True)
		inputs_train = poly.fit_transform(inputs_train)
		inputs_test = poly.fit_transform(inputs_test)

	# Scaling
	scaler_in = Scaler(inputs_train)
	inputs_train = scaler_in.transform(inputs_train)
	inputs_test = scaler_in.transform(inputs_test)

	scaler_out = Scaler(outputs_train)
	outputs_train = scaler_out.transform(outputs_train)
	outputs_test = scaler_out.transform(outputs_test)

	num_examples = len(inputs_train) #number of examples
	n_inputs = inputs_train.shape[1]
	n_outputs = outputs_train.shape[1]

	print("Examples:", num_examples)
	print("N Inputs:", n_inputs)
	print("N Outputs:", n_outputs)

	# NN variables
		#inputs
	with tf.name_scope("inputs"):
		input = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "input")
		is_training = tf.placeholder(tf.bool, shape=(), name = "is_training")
		keep_prob = tf.placeholder(tf.float32, shape=(), name = "keep_prob")

		#outputs
	with tf.name_scope("outputs"):
		output = tf.placeholder(tf.float32, shape = (None, n_outputs), name = "output")
		to_predict = tf.Variable(tf.zeros([1, n_outputs]), validate_shape=False, dtype = tf.float32, name = "to_predict")


	# NN architecture
		#regression
	hiddens = []
	w_init = tf.contrib.layers.xavier_initializer(seed=7)
	w_regularizer = tf.contrib.layers.l1_regularizer(scale=0.01)
	bn_params = {
		'is_training': is_training,
		'decay': 0.99,
		'updates_collections': None
	}
	with tf.name_scope("dnn"):
		if(N_HIDDEN == 0):
			input_drop = apply_dropout(input, keep_prob, is_training)
			regression = fully_connected(input_drop, n_outputs, scope="regression", activation_fn=None, weights_initializer=w_init)
			print(regression.name)
		else:
			for n in range(N_HIDDEN):
				input_drop = apply_dropout(input, keep_prob, is_training)
				name = "hidden" + str(n+1)
				if n == 0:
					hiddens.append(apply_dropout(fully_connected(input_drop, N_NEURONS[n], scope=name, activation_fn=tf.nn.elu, weights_initializer=w_init), keep_prob, is_training))
				else:
					hiddens.append(apply_dropout(fully_connected(hiddens[n-1], N_NEURONS[n], scope=name, activation_fn=tf.nn.elu, weights_initializer=w_init), keep_prob, is_training))

			regression = fully_connected(hiddens[N_HIDDEN-1], n_outputs, scope="regression", activation_fn=None, weights_initializer=w_init)

		#cost function
	with tf.name_scope("loss"):
		to_predict = tf.assign(to_predict, regression, name="predict") #to access in predictor
		#reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		#base_loss = tf.reduce_mean(tf.squared_difference(regression, output), name = "base_loss")
		#loss = tf.add_n([base_loss] + reg_losses, name="loss")
		loss = tf.reduce_mean(tf.squared_difference(regression, output), name = "loss")

		#optimizer
	with tf.name_scope("train"):
		#optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
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
	n_batches = num_examples // BATCH_SIZE
	model_id = int(open('model_id.txt', 'r').readline())
	generic_filename = str(model_id) + "_" + str(N_HIDDEN) + "_" + str(N_TEMPLATES) + "_" + str(CENTER) + "_" + str(POLY) + "_" + str(DROPOUT)

	curves_file = open("./Curves/" + generic_filename + ".csv", 'w')

	with tf.Session() as sess:
		init.run()
		print("---------------------STARTING TRAIN-------------------------")
		print("{:<5s} {:>12s} {:>12s}".format('EPOCH', 'TRAIN', 'TEST'))
		for epoch in range(N_EPOCHS):
			for batch_index in range(n_batches):
				x_batch, y_batch = get_next_batch(inputs_train, outputs_train, batch_index, BATCH_SIZE)
				sess.run(training_op, feed_dict = {input: x_batch, output: y_batch, keep_prob: KEEP_PROB, is_training: True})

			if epoch % CURVE_PROGRESS == 0:
				acc_train = loss.eval(feed_dict = {input: x_batch, output: y_batch, keep_prob: 1.0, is_training: False})
				acc_test = loss.eval(feed_dict = {input: inputs_test, output: outputs_test, keep_prob: 1.0, is_training: False})
				curves_file.write(str(epoch) + ',' + str(acc_train) + ',' + str(acc_test) + '\n')

				if(epoch % PRINT_PROGRESS == 0):
					print("{:<5d} {:>12f} {:>12f}".format(epoch, acc_train, acc_test))

					save_path = saver.save(sess, "./tmp/my_model.ckpt")
					summary_str = mse_summary.eval(feed_dict = {input: x_batch, output: y_batch, keep_prob: 1.0, is_training: True})
					step = epoch * n_batches + batch_index
					file_writer.add_summary(summary_str, step)

		curves_file.close()
		save_file = "./Models/" + generic_filename + ".ckpt"
		save_path = saver.save(sess, save_file)

	param_file = "./Params/" + generic_filename + ".txt"
	with open(param_file, 'w') as file:
		file.write("Metrics: {}\n".format(METRICS))
		file.write("N_HIDDEN: {}\n".format(N_HIDDEN))
		file.write("N_NEURONS: {}\n".format(N_NEURONS))
		file.write("N_TEMPLATES: {}\n".format(N_TEMPLATES))
		file.write("N_EPOCHS: {}\n".format(N_EPOCHS))
		file.write("LEARNING_RATE: {}\n".format(LEARNING_RATE))
		file.write("BATCH_SIZE: {}\n".format(BATCH_SIZE))
		file.write("DEGREE: {}\n".format(DEGREE))
		file.write("CENTER: {}\n".format(CENTER))
		file.write("POLY: {}\n".format(POLY))
		file.write("KEEP_PROB: {}\n".format(KEEP_PROB))
		file.write("DROPOUT: {}\n".format(DROPOUT))


	file_writer.close()

	with open('model_id.txt', 'w') as file:
		file.write(str(model_id+1) + '\n')

	print(datetime.utcnow().strftime("%Y%m%d%H%M%S"))

if __name__ == '__main__':
	main()
