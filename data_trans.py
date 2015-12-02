from sklearn import preprocessing
import numpy as np

#modify train_label.csv
class Preprocess_Data(object):
	def __init__(self):
		pass

	def resize_label(self, input_label):
		label_file = open(input_label)
		label_lines = label_file.readlines()
		labels = []
		for i in range(len(label_lines)):
			labels.append(int((int(label_lines[i])-0.5)*2))
		label_file.close()
		return labels

	def reformat_data(self, input_file):
		regularized_data = np.loadtxt(input_file,delimiter=',')
		normalized_data = preprocessing.normalize(regularized_data, norm='l2')
		standardized_data = preprocessing.scale(normalized_data)
		data = []
		for line in standardized_data:
			dic = {}
			order = 1
			for item in line:
				dic[order] = item
				order = order+1
			data.append(dic)
		return data
		


	
