from libsvm.python.svmutil import *
from data_trans import Preprocess_Data
from task import K_Fold_SVM
import matplotlib.pyplot as plt


p = Preprocess_Data()
labels = p.resize_label('bigger_label2.csv')
train_data = p.reformat_data('bigger_train2.csv')

t = K_Fold_SVM(train_data, labels, K=10)

best_estimate = 0
opt_param = None
opt_model = None

#test linear kernel with different cost parameters
params_set = [['-t 1 -c 100 -g 0.1 -r 10'], ['-t 1 -c 100 -d 2 -g 0.1 -r 10'], ['-t 1 -c 1000 -d 2 -g 0.1 -r 10']]

x = [1,2,3]

r_list = []
for params in params_set:
	best_model, best_param, results = t.get_optimal_model(params)
	p_label, p_acc, p_val = svm_predict(labels, train_data, best_model)
	r_list.append(p_acc[0])
	if p_acc[0] > best_estimate:
		best_estimate = p_acc[0]
		opt_model = best_model
		opt_param = best_param

plt.plot(x, r_list)
# ax = plt.gca()
# ax.set_xscale('log')
plt.xlabel('trials')
plt.ylabel('10-fold cross validation accuracy')
plt.title('results')
plt.show()


print "**************best estimate is", best_estimate
print "@@@@@@@@@@@@@@best parameter is", opt_param




