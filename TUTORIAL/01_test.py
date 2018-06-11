import matplotlib
import numpy
import scipy
import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
init = tf.global_variables_initializer() #prepare to initiazlize all variables

f = x*x*y + y + 2

with tf.Session() as sess:
	init.run()
	result = f.eval()
print(result)
