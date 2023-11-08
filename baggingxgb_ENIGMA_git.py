import sys
import numpy as np
import numpy.matlib as nb
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.svm import SVC
import scipy.io as sio
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold,GridSearchCV,cross_val_score,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as mtr
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,auc,roc_curve,mean_squared_error,balanced_accuracy_score
import itertools
from collections import Counter
from pprint import pprint
import datetime
from time import time
import warnings
warnings.filterwarnings('ignore')#warningを表示させない
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.ensemble import BalancedBaggingClassifier
import xgboost as xgb
import pickle
import numpy.matlib

# Main function
def main():
	argvs = sys.argv    # コマンドライン引数を格納したリストの取得
	argc = len(argvs)    # 引数の個数

	# Input from command-line interface
	if (argc != 2):
	    print('Usage: python baggingrcf.py ROI_data(**.csv)')
	    quit()
    
	##### Start time
	print(" ")
	startDateTime = datetime.datetime.today()
	print("Start : " + str(startDateTime))


	data=pd.read_csv(argvs[1], skipinitialspace=True)
	n_samples, n_features=data.shape



	print(sorted(Counter(data.scan_site_en).items()))
	print(sorted(Counter(data.site_en).items()))


	print('CONT: 0, NON: 1, CON: 2, UNK:3')
	for i in np.arange(0,21,1):

		print('site:%s'%sorted(Counter(data.conv_label[data.site_en==i]).items()))



	print('conv_label: %s'% sorted(Counter(data.conv_label).items()))
	print('conv_label_OG: %s'%sorted(Counter(data.Conv_Stat).items()))

	target_count = data.conv_label.value_counts()
	print('Class 0(HC):', target_count[0])
	print('Class 2 (CHR-P):', target_count[2])
	print('Proportion:', round(target_count[0] / target_count[2], 2), ': 1')
	plt.figure(1)
	target_count.plot(kind='bar', title='Count (target)')
	plt.savefig('COUNT_LABEL.png', dpi=600)



	###data of HC and CHR-T 
	#[:,'L_bankssts_surfavg':'R_insula_surfavg']
	#[:,'L_bankssts_thickavg':'R_insula_thickavg']
	#[:,'LLatVent':'ICV']

	x=data[(data.site_en!=17) &((data.conv_label==0)|(data.conv_label==2))].loc[:,'L_bankssts_surfavg':'R_insula_surfavg']
	y=data[(data.site_en!=17) &((data.conv_label==0)|(data.conv_label==2))]['Group']
	y_CL=data[(data.site_en!=17) &((data.conv_label==0)|(data.conv_label==2))]['conv_label']


	# Check

	n_samples, n_features = x.shape
	print("%d samples, %d features" % (n_samples, n_features))            
	print(" ")
	print(" HC and CHR-T data from 20 sites")
	print(sorted(Counter(y).items()))
	print(" details in CHR group")
	print(sorted(Counter(y_CL).items()))


	###validation dataset
	data_2= data[(data.site_en==17) &((data.conv_label==0)|(data.conv_label==2))].loc[:,'L_bankssts_surfavg':'R_insula_surfavg']
	y2=data[(data.site_en==17) &((data.conv_label==0)|(data.conv_label==2))]['Group']
	y2_CL=data[(data.site_en==17) &((data.conv_label==0)|(data.conv_label==2))]['conv_label']
	print(" data from site17")
	print(sorted(Counter(y2).items()))
	print(" conv label in CHR group")
	print(sorted(Counter(y2_CL).items()))

	###prediction dataset

	data_NT= data[data.conv_label==1].loc[:,'L_bankssts_surfavg':'R_insula_surfavg']
	y_NT=data[data.conv_label==1]['Group']
	y_NT_CL=data[data.conv_label==1]['conv_label']
	print(" NON-CON data to predict")
	print(sorted(Counter(y_NT).items()))
	print(" conv label in CHR group")
	print(sorted(Counter(y_NT_CL).items()))


	data_unk= data[data.conv_label==3].loc[:,'L_bankssts_surfavg':'R_insula_surfavg']
	y_unk=data[data.conv_label==3]['Group']
	y_unk_CL=data[data.conv_label==3]['conv_label']
	print(" UNK data to predict")
	print(sorted(Counter(y_unk).items()))
	print(" conv label in CHR group")
	print(sorted(Counter(y_unk_CL).items()))
    


	#split data
	#split dataset1 into train and test sets
	x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1,random_state=14, stratify=y)

	print('Training Labels: %s'%sorted(Counter(y_train).items()))
	print('Test set  Labels: %s'%sorted(Counter(y_test).items()))
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=100,random_state=27)

	#standardalization
	sc=StandardScaler()


	clf = xgb.XGBClassifier(use_label_encoder=False,booster='gbtree',eval_metric='logloss')

	#undersampling+bagging

	bg_clf=BalancedBaggingClassifier(base_estimator=clf, sampling_strategy='majority' )

	pipe = Pipeline(steps=[
                    ('preprocessing',sc),
                    ('bg_clf', bg_clf)
                     ])


	#parameters used for grid search
	param_grid = [{
	                'preprocessing':[sc,None],
	                'bg_clf__base_estimator__learning_rate':[0.1],#0.1
	                'bg_clf__base_estimator__max_depth' :[2,3,5], #3
	                'bg_clf__base_estimator__n_estimators' : [100,200,300],#300
	                'bg_clf__base_estimator__subsample':[0.7],#0.7
	                'bg_clf__base_estimator__colsample_bytree':[0.7]#0.7

					}]


	# パラメータチューニングをグリッドサーチ
	grid_bgxg = GridSearchCV(estimator = pipe,
	                           param_grid = param_grid,
	                           scoring = 'accuracy', #accuracy, balanced_accuracy
	                           cv = cv,
	                           return_train_score = True,
	                           n_jobs = 6)



	print("Performing grid search...")
	print("parameters:")
	pprint(param_grid)
	t0 = time()
	grid_bgxg.fit(x_train, y_train)
	print("done in %0.3fs" % (time() - t0))
	print()


	filename = 'baggingxgb.sav'
	pickle.dump(grid_bgxg, open(filename, 'wb'))

	# ベストな分類器を抽出
	pprint(grid_bgxg.best_estimator_)
	# ベストなパラメータを抽出
	pprint(grid_bgxg.best_params_)
	# ベストな正解率を抽出
	pprint(grid_bgxg.best_score_)
	# 各種途中結果を抽出
	pprint(grid_bgxg.cv_results_)



	# Prediction using test set
	pred = grid_bgxg.predict(x_test)
	score=grid_bgxg.score(x_test, y_test)
	print('Test set score: {}'.format(score))
	print(confusion_matrix(y_test, pred))
	print(classification_report(y_test, pred))

	data_2=np.array(data_2)
	# Prediction using data from site17
	score2=grid_bgxg.score(data_2,y2)
	print('cv score from GV:{}'.format(score2))


	pred_2 = grid_bgxg.predict(data_2)
	print(confusion_matrix(y2, pred_2))
	print(classification_report(y2, pred_2))

	# Prediciton using non-converted & unk data
	print("prected label for NON")
	data_NT=np.array(data_NT)
	pred_y_NT = grid_bgxg.predict(data_NT)
	print(pred_y_NT)

	print("predicted label of UNK")
	data_unk=np.array(data_unk)
	pred_y_unk = grid_bgxg.predict(data_unk)
	print(pred_y_unk)


	##### Visualization
	class_names = np.array(['HC','CHR-P'])
	#Confusion matrix figure
	def plot_confusion_matrix(cm, classes,
	                          normalize=False,
	                          title='Confusion matrix',
	                          cmap=plt.cm.Blues):
	    
	    if normalize:
	        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	        print("Normalized confusion matrix")
	    else:
	        print('Confusion matrix, without normalization')
	    
	    print(cm)
	    
	    plt.imshow(cm, interpolation='nearest', cmap=cmap)
	    plt.title(title)
	    plt.colorbar()
	    tick_marks = np.arange(len(classes))
	    plt.xticks(tick_marks, classes, rotation=45)
	    plt.yticks(tick_marks, classes)
	    
	    fmt = '.2f' if normalize else 'd'
	    thresh = cm.max() / 2.
	    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	        plt.text(j, i, format(cm[i, j], fmt),
	                 horizontalalignment="center",
	                 color="white" if cm[i, j] > thresh else "black")
	    
	    plt.tight_layout()
	    plt.ylabel('True label')
	    plt.xlabel('Predicted label')

	# Compute confusion matrix
	cnf_matrix = confusion_matrix(y_test, y_pred_test)
	np.set_printoptions(precision=2)
	cnf_matrix2 = confusion_matrix(y2, pred_2)
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plt.figure(2)
	plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
	                      title='Confusion matrix, without normalization')
	plt.tight_layout()
	plt.savefig('confusion heatmap.png', dpi=600)
	# Plot normalized confusion matrix
	plt.figure(3)
	plot_confusion_matrix(cnf_matrix2, classes=class_names, normalize=True,
	                                    title='Normalized confusion matrix')
	plt.tight_layout()
	plt.savefig('confusion heatmap2.png', dpi=600)



	#feature importance
	feature_importances = np.mean([xgb.named_steps['classifier'].feature_importances_
	    for xgb in grid_bgrf.best_estimator_.named_steps['bg_clf'].estimators_], axis=0)



	all_DF=pd.DataFrame(feature_importances.reshape(1,-1), columns=x_train.columns, index=['weights']).T
	all_DF=all_DF.rename_axis('feature_names')
	all_DF=all_DF.reset_index()



	#plot feature importance   
	plt.figure(4, figsize=(10,10))
	f_data=all_DF
	f_data=f_data.sort_values(by='weights',ascending=False)
	f_data.to_csv('features_weights.csv', index=False)
	by_feature=sns.barplot(x="feature_names", y="weights", data=f_data)

	for item in by_feature.get_xticklabels():
	    item.set_rotation(90)
	    item.set_fontsize(5)
	plt.savefig("feature_importance.png",format = 'png', dpi = 600)


	##### End time
	print(" ")
	endDateTime = datetime.datetime.today()
	print("End : " + str(endDateTime))
	print("Calculation time : " + str(endDateTime - startDateTime))
	print(" ")


# The first function to be run
if __name__ == "__main__":
    main()
