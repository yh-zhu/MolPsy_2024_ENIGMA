#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# start : Sept 7, 2019
# finish : , 2019
#This coede is for harmonized 3 datasets.
# PCA +SVC (Gridsearch) + cross validation(3rd dataset)


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
    warnings.filterwarnings('ignore')#warningを表示させない
    argvs = sys.argv    # コマンドライン引数を格納したリストの取得
    argc = len(argvs)    # 引数の個数

    # Input from command-line interface
    if (argc != 3):
        print('Usage: python my_permutation.py harmonized_data(**.npy) saved_classifier(**.sav)')
        quit()


    ##### Start time
    print(" ")
    startDateTime = datetime.datetime.today()
    print("Start : " + str(startDateTime))



    ##### Read input data from files

    # File open

    data=pd.read_csv(argvs[1], skipinitialspace=True)
    grid_bgxg= pickle.load(open(argvs[2], 'rb'))



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



    ###use the best score doing permutation test(label shuffle)
    from sklearn.utils import indexable, check_random_state, _safe_indexing
    from sklearn.utils.metaestimators import _safe_split
    from sklearn.model_selection import check_cv
    from sklearn.base import is_classifier, clone
    from sklearn.metrics import check_scoring
    from joblib import Parallel, delayed



    def my_permutation_test_score(my_best_score, estimator, X, y, *, groups=None, cv=None,
                               n_permutations=100, n_jobs=None, random_state=0,
                               verbose=0, scoring=None):
        """Evaluate the significance of a cross-validated score with permutations
        Read more in the :ref:`User Guide <cross_validation>`.
        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.
        X : array-like of shape at least 2D
            The data to fit.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
            The target variable to try to predict in the case of
            supervised learning.
        groups : array-like of shape (n_samples,), default=None
            Labels to constrain permutation within groups, i.e. ``y`` values
            are permuted among samples with the same group identifier.
            When not specified, ``y`` values are permuted among all samples.
            When a grouped cross-validator is used, the group labels are
            also passed on to the ``split`` method of the cross-validator. The
            cross-validator uses them for grouping the samples  while splitting
            the dataset into train/test set.
        scoring : str or callable, default=None
            A single str (see :ref:`scoring_parameter`) or a callable
            (see :ref:`scoring`) to evaluate the predictions on the test set.
            If None the estimator's score method is used.
        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a `(Stratified)KFold`,
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.
            For int/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, :class:`StratifiedKFold` is used. In all
            other cases, :class:`KFold` is used.
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.
            .. versionchanged:: 0.22
                ``cv`` default value if None changed from 3-fold to 5-fold.
        n_permutations : int, default=100
            Number of times to permute ``y``.
        n_jobs : int, default=None
            The number of CPUs to use to do the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        random_state : int, RandomState instance or None, default=0
            Pass an int for reproducible output for permutation of
            ``y`` values among samples. See :term:`Glossary <random_state>`.
        verbose : int, default=0
            The verbosity level.
        Returns
        -------
        score : float
            The true score without permuting targets.
        permutation_scores : array of shape (n_permutations,)
            The scores obtained for each permutations.
        pvalue : float
            The p-value, which approximates the probability that the score would
            be obtained by chance. This is calculated as:
            `(C + 1) / (n_permutations + 1)`
            Where C is the number of permutations whose score >= the true score.
            The best possible p-value is 1/(n_permutations + 1), the worst is 1.0.
        Notes
        -----
        This function implements Test 1 in:
            Ojala and Garriga. Permutation Tests for Studying Classifier
            Performance.  The Journal of Machine Learning Research (2010)
            vol. 11
            `[pdf] <http://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf>`_.
        """

        def _permutation_test_score(estimator, X, y, groups, cv, scorer):
            """Auxiliary function for permutation_test_score"""
            avg_score = []
            for train, test in cv.split(X, y, groups):
                X_train, y_train = _safe_split(estimator, X, y, train)
                X_test, y_test = _safe_split(estimator, X, y, test, train)
                estimator.fit(X_train, y_train)
                avg_score.append(scorer(estimator, X_test, y_test))
            return np.mean(avg_score)


        def _shuffle(y, groups, random_state):
            """Return a shuffled copy of y eventually shuffle among same groups."""
            if groups is None:
                indices = random_state.permutation(len(y))
            else:
                indices = np.arange(len(groups))
                for group in np.unique(groups):
                    this_mask = (groups == group)
                    indices[this_mask] = random_state.permutation(indices[this_mask])
            return _safe_indexing(y, indices)
        
        
            X, y, groups = indexable(X, y, groups)

        cv = check_cv(cv, y, classifier=is_classifier(estimator))
        scorer = check_scoring(estimator, scoring=scoring)
        random_state = check_random_state(random_state)

        # We clone the estimator to make sure that all the folds are
        # independent, and that it is pickle-able.
        score = my_best_score
        permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_permutation_test_score)(
                clone(estimator), X, _shuffle(y, groups, random_state),
                groups, cv, scorer)
            for _ in range(n_permutations))
        permutation_scores = np.array(permutation_scores)
        pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1)
        return score, permutation_scores, pvalue

    n_classes = np.unique(y_train).size
    score, permutation_scores, pvalue = my_permutation_test_score(grid_bgxg.best_score_,
        grid_bgxg, x_train, y_train, scoring="accuracy", cv=10, n_permutations=1000, n_jobs=1)

    np.save("permutation_score.npy", permutation_scores)
    print("Classification score %s (pvalue : %s)" % (score, pvalue))

    # #############################################################################
    # View histogram of permutation scores
    plt.figure()
    plt.rcParams['font.family'] ='Arial'
    plt.rcParams["font.size"] = 17
    plt.hist(permutation_scores, 20, label='Permutation scores',
             edgecolor='black')
    ylim = plt.ylim()
    # BUG: vlines(..., linestyle='--') fails on older versions of matplotlib
    # plt.vlines(score, ylim[0], ylim[1], linestyle='--',
    #          color='g', linewidth=3, label='Classification Score'
    #          ' (pvalue %s)' % pvalue)
    # plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',
    #          color='k', linewidth=3, label='Luck')
    plt.plot(2 * [score], ylim, '--g', linewidth=3,
             label='Classification Score'
             ' (pvalue %s)' % pvalue)
    plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

    plt.ylim(ylim)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Score')
    plt.savefig('permutation_test.png', bbox_inches='tight', dpi=600)                  
                                                  
    # parameters
    print("Best parameters ")

    print(grid_bgxg.best_params_)
                          
                          
                          
    ##### End time
    print(" ")
    endDateTime = datetime.datetime.today()
    print("End : " + str(endDateTime))
    print("Calculation time : " + str(endDateTime - startDateTime))
    print(" ")



# The first function to be run
if __name__ == "__main__":
    main()

