import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


n_epochs = 1000
learning_rate = 0.01

#get data
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

#normalize data
scaler = StandardScaler()
scaler.fit(housing.data)
scaled_housing = scaler.transform(housing.data)

#add bias
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing.data]

x = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="x")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(x, theta, name="predictions")
error = y_pred-y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) #change this to implement different optimizers like momentum gradient descent
training_op = optimizer.minimize(mse)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(n_epochs):
		if epoch % 100 == 0:
			print("Epoch", epoch, "MSE = ", mse.eval())
			save_path = saver.save(sess, "/tmp/my_model.ckpt") #este path fica em C:/tmp/

		sess.run(training_op)

	best_theta = theta.eval()
	save_path = saver.save(sess, "/tmp/my_model_final.ckpt") #este path fica em C:/tmp/

'''
To restore a session check pág 296
'''
