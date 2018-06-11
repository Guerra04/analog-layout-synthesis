from Scaler import Scaler
from sklearn.preprocessing import StandardScaler
import dataset_processing as proc
import dataset_dependent as dd

data_train = proc.read_file("dataset_train.csv")

inputs_train = dd.get_inputs(data_train)

scaler_in1 = StandardScaler()
scaler_in1.fit(inputs_train)
scaler_in2 = Scaler(inputs_train)

inputs_train1 = scaler_in1.transform(inputs_train)
inputs_train2 = scaler_in2.transform(inputs_train)

for i in range(len(inputs_train)):
	print(inputs_train1[i, 1], "|||", inputs_train2[i, 1])
