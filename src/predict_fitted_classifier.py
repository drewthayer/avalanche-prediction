from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pdb

from transformation_scripts import set_df_index_from_datetime
from sqlite3_scripts import connect_to_sql
from modeling_scripts import train_test_split_time, train_estimator
from modeling_scripts import predict_classifier, print_scores

def multi_case_classifier_predict(df, cases, ests, stdzrs,
                n_oversamps, c_true, c_pred):
    y_true_l = []
    y_hat_l = []
    y_proba_l = []
    feats_l = []

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    for case, est, stdzr, n, c_t, c_p in zip(cases, ests, stdzrs,
            n_oversamps, c_true, c_pred):
        data_df = df.copy() # copy to read all columns after dropping
        print('case: {}'.format(case[0]))

        # drop other binary and probability column
        c_drop = [c for c in list(df.columns) if case[1] in c]
        data_df.drop(c_drop, axis=1, inplace=True)

        # train test split in time
        X_train, y_train, X_test, y_test = train_test_split_time(data_df,
            '2016-06-01', case[0])

        # datetime for plot
        test_datetime = pd.to_datetime(X_test.index)

        ''' predict with fitted model  '''
        y_hat, y_proba, importances = predict_classifier(X_test, y_test,
                est, stdzr)

        # save true, predicted, proba, feats
        y_true_l.append(y_test)
        y_hat_l.append(y_hat)
        y_proba_l.append(y_proba)
        feats_l.append((X_train.columns,importances))

        # print scores
        method_list = [accuracy_score, recall_score, precision_score]
        print_scores(y_test, y_hat, method_list)

    return y_true_l, y_hat_l, y_proba_l, feats_l, list(y_test.index)


if __name__=='__main__':
    zonename = sys.argv[1]
    # load data from sql as df, drop n_avy column
    current = os.getcwd()
    conn = connect_to_sql(current + '/../data/data-engineered.db')
    df = pd.read_sql("select * from {}".format(zonename), conn)

    # drop n_avy column and set index to datetime
    df.drop('N_AVY', axis=1, inplace=True)
    df = set_df_index_from_datetime(df, 'datetime')

    # fill na with zero in case any not imputed
    df.fillna(0, inplace=True)

    #load fitted models
    slabfile = 'best-ests/{}_best_est_gbc_SLAB.p'.format(zonename)
    wetfile = 'best-ests/{}_best_est_gbc_WET.p'.format(zonename)
    est_slab, std_slab = pickle.load( open( slabfile, 'rb'))
    est_wet, std_wet = pickle.load( open( wetfile, 'rb'))

    ''' N_AVY when case = slab/wet '''
    params = {
    'cases': [['SLAB','WET'], ['WET','SLAB']],
    'ests': [est_slab, est_wet],
    'stdzrs': [std_slab, std_wet],
    'c_true': ['b','g'],
    'c_pred': ['r','orange'],
    'n_oversamps': [1,1]}

    outputs = multi_case_classifier_predict(df, **params)

    # save outputs to pkl
    pickle.dump(outputs,
            open('outputs/{}_gbc_output.p'.format(zonename),'wb'))
