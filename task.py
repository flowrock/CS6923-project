from libsvm.python.svmutil import *
import random
import sys

class K_Fold_SVM(object):
	def __init__(self, train_data, labels, K):
		self.K = K
		if self.K <= 0:
			sys.exit('K should be positive!')
		self.total_y, self.total_x = labels, train_data
		self.fold_size = len(self.total_y)/self.K
		if self.fold_size < 2:
			sys.exit('K is too big!')

	def get_optimal_param(self, cand_params):
		best_accuracy = 0
		best_param = None
		for param in cand_params:
			r = self.cross_validation(param)
			if r>best_accuracy:
				best_accuracy = r
				best_param = param
		print 'best accuracy is', best_accuracy
		print 'best parameter is', best_param
		return best_param

	def cross_validation(self, param):
		folds_x, folds_y = self.get_k_folds()
		acc_total = 0
		for i in range(0,len(folds_x)):
			test_fold_x = folds_x[i]
			test_fold_y = folds_y[i]
			train_fold_x = []
			train_fold_y = []
			for j in range(0,len(folds_x)):
				if j!=i:
					train_fold_x.extend(folds_x[j])
					train_fold_y.extend(folds_y[j])
			prob = svm_problem(train_fold_y, train_fold_x)
			train_param = svm_parameter(param)
			m = svm_train(prob, train_param)
			p_label, p_acc, p_val = svm_predict(test_fold_y, test_fold_x, m)
			print "round", i+1
			acc_total = acc_total+p_acc[0]
		acc_avg = acc_total/self.K
		print "***********average accuracy is", acc_avg
		return acc_avg

	def get_k_folds(self):
		copy_x = self.total_x[:]
		copy_y = self.total_y[:]
		folds_x = []
		folds_y = []
		while len(copy_x) >= self.fold_size:
			curr_fold_x = []
			curr_fold_y = []
			while len(curr_fold_x)<self.fold_size:
				pos = random.randint(0,len(copy_x)-1)
				curr_fold_x.append(copy_x[pos])
				curr_fold_y.append(copy_y[pos])
				del copy_x[pos]
				del copy_y[pos]
			folds_x.append(curr_fold_x)
			folds_y.append(curr_fold_y)
		folds_x[-1].extend(copy_x)
		folds_y[-1].extend(copy_y)
		return folds_x, folds_y

	def get_optimal_model(self, cand_params):
		best_param = self.get_optimal_param(cand_params)
		m = svm_train(self.total_y, self.total_x, best_param)
		return m, best_param


