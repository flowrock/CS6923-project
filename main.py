from libsvm.python.svmutil import *
from data_trans import Preprocess_Data
from task import K_Fold_SVM

p = Preprocess_Data()
labels = p.resize_label('bigger_label.csv')
train_data = p.reformat_data('bigger_train.csv')

t = K_Fold_SVM(train_data, labels, K=10)

best_estimate = 0
opt_param = None
opt_model = None

#test linear kernel with different cost parameters
params_set = [['-t 0 -c 0.01', '-t 0 -c 0.1', '-t 0 -c 1', '-t 0 -c 10', '-t 0 -c 100', '-t 0 -c 1000'],
		['-t 1 -c 0.01', '-t 1 -c 0.1', '-t 1 -c 1', '-t 1 -c 10', '-t 1 -c 100', '-t 1 -c 1000'],
		['-t 2 -c 0.01', '-t 2 -c 0.1', '-t 2 -c 1', '-t 2 -c 10', '-t 2 -c 100', '-t 2 -c 1000']]

for params in params_set:
	best_model, best_param = t.get_optimal_model(params)

	p_label, p_acc, p_val = svm_predict(labels, train_data, best_model)
	if p_acc[0] > best_estimate:
		best_estimate = p_acc[0]
		opt_model = best_model
		opt_param = best_param

print "**************best estimate is", best_estimate
print "@@@@@@@@@@@@@@best parameter is", opt_param




