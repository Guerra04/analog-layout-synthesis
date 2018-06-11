import tensorflow as tf
from constants import *

class Rectangle:
	def __init__(self, x, y, width, height):
		self.xmin = x
		self.ymin = y
		self.xmax = x + width
		self.ymax = y + height

'''
# TF version
def overlap_area(xmin_a, ymin_a, xmax_a, ymax_a, xmin_b, ymin_b, xmax_b, ymax_b):
	dx = tf.subtract(tf.minimum(xmax_a, xmax_b), tf.maximum(xmin_a, xmin_b))
	dy = tf.subtract(tf.minimum(ymax_a, ymax_b), tf.maximum(ymin_a, ymin_b))
	#result = tf.cond(dx >= 0 and dy >= 0, lambda: tf.multiply(dx, dy), lambda: 0)
	area = tf.multiply(dx, dy)

	aux1 = tf.greater_equal(dx, tf.zeros(dx.shape))
	aux2 = tf.greater_equal(dy, tf.zeros(dy.shape))
	cond = tf.logical_and(aux1, aux2)
	result = tf.where(cond, area, tf.zeros_like(area))
	return result
'''

def overlap_area(a, b):
	dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
	dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)

	if(dx >= 0 and dy >= 0):
		return dx*dy
	else:
		return 0

'''
#TF version
def overlap_array(regression, inputs_batch):
	overlap = tf.constant(0.0)
	i = tf.constant(0)
	def cond1(i, overlap):
		return tf.less(i, N_DEVICES)
	def body1(i, overlap):
		a = Rectangle(regression[:, 2*i], regression[:, 1+2*i], inputs_batch[:, 3+5*i], inputs_batch[:, 4+5*i])

		j = tf.constant(i)
		def cond2(j, overlap):
			return tf.less(j, N_DEVICES)
		def body2(j, overlap):
			b = Rectangle(regression[:, 2*j], regression[:, 1+2*j], inputs_batch[:, 3+5*j], inputs_batch[:, 4+5*j])
			overlaps = tf.py_func(overlap_area, [a,b], [tf.float32])
			overlap = tf.add(overlap, overlaps)
			return tf.add(j, 1), overlap

		j, overlap = tf.while_loop(cond2, body2, (j, overlap))
		return tf.add(i, 1), overlap

	i, overlap_result = tf.while_loop(cond1, body1, (i, overlap))
	return overlap_result
'''

def compute_overlap(regression, inputs_batch, debug=False):
	overlap = 0.0
	for i in range(N_DEVICES):
		a = Rectangle(regression[2*i], regression[1+2*i], inputs_batch[3+5*i], inputs_batch[4+5*i])
		for j in range(i+1, N_DEVICES):
			b = Rectangle(regression[2*j], regression[1+2*j], inputs_batch[3+5*j], inputs_batch[4+5*j])
			overlaps = overlap_area(a, b)
			overlap = overlap + overlaps
			if(debug):
				print(i, "-", j, ":", overlaps)

	return overlap
