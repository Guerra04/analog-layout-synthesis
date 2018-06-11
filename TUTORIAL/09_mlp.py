import tensorflow as tf

def neuron_layer(x, n_neurons, name, activation=None):
	with tf.name_scope(name):
		n_inputs = int(x.get_shape()[1])
		stddev = 2/np.sqrt(n_inputs)
		init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
		W = tf.Variable(init, name="Weights")
		b = tf.Variable(tf.zeros([n_neurons]), name="biases")
		z = tf.matmul(x,W) + b
		if activation=="relu":
			return tf.nn.relu(z)
		else:
			return z

n_inputs = 28*28 #MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

x = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "x")
y = tf.placeholder(tf.int64, shape = (None), name = "y")

with tf.name_scope("dnn"):
	hidden1 = neuron_layer(x, n_hidden1, "hidden1", activation="relu")
	hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
	logits = neuron_layer(hidden2, n_outputs, "outputs")

	
