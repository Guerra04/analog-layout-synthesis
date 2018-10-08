import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv('Curves/53_3_2_1_1_0.csv', names=['epochs', 'train', 'test'])

plt.plot(data['epochs'], data['train'], 'r', label='Training Batch')
plt.plot(data['epochs'], data['test'], 'b', label='Validation Set')
plt.legend(loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()
