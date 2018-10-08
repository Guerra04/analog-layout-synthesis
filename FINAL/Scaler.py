import numpy as np

class Scaler:
	def __init__(self, array):
		self.mean = self.comp_mean(array)
		self.std = self.comp_std(array)

	def comp_mean(self, array):
		mean = np.zeros(array.shape[1])
		for col in range(array.shape[1]):
			mean[col] = np.mean(array[:,col])
		return mean

	def comp_std(self, array):
		th = 1e-10
		std = np.zeros(array.shape[1])
		for col in range(array.shape[1]):
			std[col] = np.std(array[:,col])
			if std[col] < th:
				std[col] = th

		return std

	def transform(self, array):
		transformation = np.zeros(array.shape)
		for col in range(array.shape[1]):
			for line in range(array.shape[0]):
				transformation[line, col] = (array[line, col] - self.mean[col])/self.std[col]

		return transformation

	def inverse_transform(self, array):
		transformation = np.zeros(array.shape)
		for col in range(array.shape[1]):
			for line in range(array.shape[0]):
				transformation[line, col] = array[line, col]*self.std[col] + self.mean[col]

		return transformation
