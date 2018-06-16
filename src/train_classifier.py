from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd
import pickle
import numpy as np
import os
import sys

from scripts.transformation_scripts import smote, set_df_index_from_datetime
from scripts.sqlite3_scripts import connect_to_sql
from scripts.modeling_scripts import train_test_split_time, train_estimator
from scripts.modeling_scripts import predict_classifier, print_scores

def multi_case_classifier_train(df, cases, case_params, case_select='slab'):
    ''' case: list, length = # of cases, items = list of [a,b] and [b,a]
        case_params, list, same length as case, items = dicts of params
        '''
    if case_select == 'slab':
        case = cases[0]
        params = case_params[0]
    elif case_select == 'wet':
        case = cases[1]
        params = case_params[1]

    data_df = df.copy() # copy to read all columns after dropping

    # drop other binary and probability column
    c_drop = [c for c in list(df.columns) if case[1] in c]
    data_df.drop(c_drop, axis=1, inplace=True)

    # train test split in time
    X_train, y_train, X_test, y_test = train_test_split_time(data_df,
        '2016-06-01', case[0])

    # oversample train data
    X_smoted, y_smoted = smote(X_train.values, y_train.values, 0.60, k=None)

    # train model
    model = GradientBoostingClassifier()
    # train
    est, stndzr = train_estimator(model, params, X_smoted, y_smoted, standardize=True)
    # training cv score
    n_folds= 10
    score_method = 'recall'
    scores = cross_validate(est, X_smoted, y_smoted, scoring=score_method,
        cv=n_folds, n_jobs=-1, verbose=1)
    cv_score = scores['test_score'].mean()
    print('{} fold cv train score ({}) = {:0.3f}'.format(n_folds,
            score_method, cv_score))
    # predict
    y_hat, y_proba, importances = predict_classifier(X_test, y_test, est, stndzr)
    # print test scores
    method_list = [accuracy_score, recall_score, precision_score]
    print('case: {}'.format(case[0]))
    print_scores(y_test, y_hat, method_list)

    return est, stndzr, case[0]

'''
Trains a classifer with parameters and writes a fitted estimator to pickle

inputs:

.pkl file with tranformed, engineered data

outputs:

.pkl file: tuple(fitted estimator, fitted standardizer)

'''

if __name__=='__main__':
    zonename = sys.argv[1] # 'aspen' or 'nsj'
    case_select = sys.argv[2] # 'slab' or 'wet'
    # load data from sql as df, drop n_avy column
    current = os.getcwd()
    conn = connect_to_sql(current + '/../data/data-engineered.db')
    df = pd.read_sql("select * from {}".format(zonename), conn)

    # drop n_avy column and set index to datetime
    df.drop('N_AVY', axis=1, inplace=True)
    df = set_df_index_from_datetime(df, 'datetime')

    # fill na with zero in case any not imputed
    df.fillna(0, inplace=True)

    ''' case : slab or wet '''
    cases = [['SLAB','WET'], ['WET','SLAB']]

    case_params = [{'criterion': 'friedman_mse',
        'learning_rate': 0.01,
        'loss': 'exponential',
        'max_features': 'log2',
        'min_samples_leaf': 4,
        'min_samples_split': 6,
        'n_estimators': 400,
        'subsample': 0.8,
        'verbose': 1},

        {'criterion': 'friedman_mse',
            'learning_rate': 0.05,
            'loss': 'deviance',
            'max_features': 'log2',
            'min_samples_leaf': 5,
            'min_samples_split': 5,
            'n_estimators': 600,
            'subsample': 0.4,
            'verbose': 1}]
    est, stndzr, name = multi_case_classifier_train(df, cases, case_params, case_select)
    # save model and standarizer to pickle
    pickle.dump((est, stndzr), open("best-ests/{}_best_est_gbc_{}.p".format(zonename, name), "wb"))
