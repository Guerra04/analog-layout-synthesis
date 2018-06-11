import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data #dataset MNIST

mnist = input_data.read_data_sets("/tmp/data/")
n_inputs = 28*28 #MNIST

#Neural network architecture
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#training parameters
learning_rate = 0.01
n_epochs = 40
batch_size = 50

x = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "x")
y = tf.placeholder(tf.int64, shape = (None), name = "y")

#definition of the network
with tf.name_scope("dnn"):
	hidden1 = fully_connected(x, n_hidden1, scope="hidden1")
	hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
	logits = fully_connected(hidden2, n_outputs, scope="outputs",
				activation_fn=None)

#cost function
with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=y, logits=logits)
	loss = tf.reduce_mean(xentropy, name="loss")

#learning step
with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_op = optimizer.minimize(loss)

#evaluate the model
with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1) #check if prediction is correct
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) #mean accuracy

#initialization
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#create session
with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for iter in range(mnist.train.num_examples // batch_size):
			x_batch, y_batch = mnist.train.next_batch(batch_size)
			sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict={x: x_batch, y: y_batch})
		acc_test = accuracy.eval(feed_dict={x: mnist.test.images,
					y: mnist.test.labels})

		print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

	save_path = saver.save(sess, "./my_model_final.ckpt")


'''
To make predictions:

with tf.Session() as sess:
	saver.restore(sess, "./my_model_final.ckpt")
	x_new_scaled = [...] #new images (scaled from 0 to 1)
	z = logits.eval(feed_dict={x: x_new_scaled})
	y_pred = np.argmax(z, axis=1)
'''
