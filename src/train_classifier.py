from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd
import pickle
#import matplotlib.pyplot as plt
import numpy as np

from transformation_scripts import div_count_pos_neg, smote

def print_scores(y_true, y_hat, method_list):
    methods = [accuracy_score, recall_score, precision_score]
    for method in methods:
        score = method(y_test, y_hat)
        print('test {} = {:0.3f}'.format(method.__name__, score))

def train_estimator(est, params, X_train, y_train):
    est.fit(X_train, y_train)
    print('oob score: {:0.3f}'.format(est.oob_score))
    return est

def predict_classifier(X_test, y_test, est):
    ''' applies .predict() method of fitted classifier to X,y data '''
    y_hat = est.predict(X_test)
    y_proba = est.predict_proba(X_test)
    importances = est.feature_importances_
    return y_hat, y_proba, importances

def print_scores(y_true, y_hat, method_list):
    for method in method_list:
        score = method(y_true, y_hat)
        print('test {} = {:0.3f}'.format(method.__name__, score))


if __name__=='__main__':
    # load data
    df = pickle.load( open( 'pkl/aspen_d2_imputemean_alldays.p', 'rb'))
    df.drop('N_AVY', axis=1, inplace=True)
    # fill na with zero in case any not imputed
    df.fillna(0, inplace=True)

    ''' case : slab or wet '''
    cases = [['SLAB','WET'], ['WET','SLAB']]

    ''' run case   '''
    case = cases[0] # chose case here, 0 or 1
    data_df = df.copy() # copy to read all columns after dropping

    # drop other binary and probability column
    c_drop = [c for c in list(df.columns) if case[1] in c]
    data_df.drop(c_drop, axis=1, inplace=True)

    # train test split in time
    splitdate = pd.to_datetime('2016-06-01')
    train_df = data_df[data_df.index <= splitdate]
    test_df = data_df[data_df.index > splitdate]

    ''' select features and target X,y '''
    # train set
    X_train = train_df.copy()
    y_train = X_train.pop(case[0])
    # test set
    X_test = test_df.copy()
    y_test = X_test.pop(case[0])

    # oversample train data
    X_smoted, y_smoted = smote(X_train.values, y_train.values, 0.60, k=None)

    # train model
    params = {'criterion': 'gini',
        'max_features': 'log2',
        'min_samples_leaf': 10,
        'min_samples_split': 8,
        'n_estimators': 500,
        'n_jobs': -1,
        'oob_score': True,
        'verbose': 1}

    model = RandomForestClassifier()

    # train
    est = train_estimator(model, params, X_smoted, y_smoted)
    # predict
    y_hat, y_proba, importances = predict_classifier(X_test, y_test, est)
    # print scores
    method_list = [accuracy_score, recall_score, precision_score]
    print('case: {}'.format(case[0]))
    print_scores(y_test, y_hat, method_list)

    #pickle.dump(best_est, open("best-ests/best_est_gbc_{}.p".format(case[0]), "wb"))
