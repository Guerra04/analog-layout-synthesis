import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def fetch_batch(epoch, batch_index, batch_size, data, target):
	first = batch_index*batch_size
	last = batch_index*batch_size + batch_size

	x_batch64 = data[first : last, :]
	y_batch64 = target[first : last]

	x_batch = x_batch64.astype(np.float32)
	y_batch_1D = y_batch64.astype(np.float32)

	y_batch = y_batch_1D.reshape(-1, 1)

	return x_batch, y_batch

#get data
housing = fetch_california_housing()
m, n = housing.data.shape

#learning variables
n_epochs = 1000
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(m/batch_size))

#normalize data
scaler = StandardScaler()
scaler.fit(housing.data)
scaled_housing = scaler.transform(housing.data)

#add bias
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing.data]

#tf variables
x = tf.placeholder(tf.float32, shape=(None, n+1), name="x")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(x, theta, name="predictions")
error = y_pred-y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) #change this to implement different optimizers like momentum gradient descent
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()


xx, yy = fetch_batch(0, 0, batch_size, scaled_housing_data_plus_bias, housing.target)
print(xx.dtype)

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			x_batch, y_batch = fetch_batch(epoch, batch_index, batch_size, scaled_housing_data_plus_bias, housing.target)
			sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
		#if epoch % 100 == 0:
		#	print("Epoch", epoch, "MSE =", mse.eval())
	best_theta = theta.eval()
