import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

x = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="Y")
xT = tf.transpose(x)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(xT, x)), xT), y)

with tf.Session() as sess:
	theta_value = theta.eval()
	print(theta)
