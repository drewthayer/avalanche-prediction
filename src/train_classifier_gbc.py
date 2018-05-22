from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd
import pickle
#import matplotlib.pyplot as plt
import numpy as np

from transformation_scripts import div_count_pos_neg, smote

'''
Trains a classifer with parameters and writes a fitted estimator to pickle
'''

def train_test_split_time(df, splitdate_str, y_col):
    splitdate = pd.to_datetime(splitdate_str)
    train_df = df[df.index <= splitdate]
    test_df = df[df.index > splitdate]

    # train set
    X_train = train_df.copy()
    y_train = X_train.pop(y_col)
    # test set
    X_test = test_df.copy()
    y_test = X_test.pop(y_col)

    return X_train, y_train, X_test, y_test

def train_estimator(est, params, X_train, y_train, standardize=True):
    if standardize:
        pipe = Pipeline([
            ('standardizer', StandardScaler()),
            ('est', est)])
        pipe.fit(X_train, y_train)
        return pipe.named_steps['est'], pipe.named_steps['standardizer']
    else:
        est.fit(X_train, y_train)
        return est, None

def predict_classifier(X_test, y_test, est, standardizer):
    ''' applies .predict() method of fitted classifier to X,y data '''
    X_scaled = standardizer.transform(X_test)
    y_hat = est.predict(X_scaled)
    y_proba = est.predict_proba(X_scaled)
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
    #df.drop('MONTH', axis=1, inplace=True)
    # fill na with zero in case any not imputed
    df.fillna(0, inplace=True)

    ''' case : slab or wet '''
    cases = [['SLAB','WET'], ['WET','SLAB']]

    ''' run case   '''
    case = cases[1] # chose case here, 0 or 1
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
    params_slab = {'criterion': 'friedman_mse',
        'learning_rate': 0.01,
        'loss': 'exponential',
        'max_features': 'log2',
        'min_samples_leaf': 4,
        'min_samples_split': 6,
        'n_estimators': 400,
        'subsample': 0.8,
        'verbose': 1}

    params_wet = {'criterion': 'friedman_mse',
        'learning_rate': 0.05,
        'loss': 'deviance',
        'max_features': 'log2',
        'min_samples_leaf': 5,
        'min_samples_split': 5,
        'n_estimators': 600,
        'subsample': 0.4,
        'verbose': 1}

    model = GradientBoostingClassifier()
    # train
    est, stndzr = train_estimator(model, params_wet, X_smoted, y_smoted, standardize=True)
    # predict
    y_hat, y_proba, importances = predict_classifier(X_test, y_test, est, stndzr)
    # print scores
    method_list = [accuracy_score, recall_score, precision_score]
    print('case: {}'.format(case[0]))
    print_scores(y_test, y_hat, method_list)

    pickle.dump((est, stndzr), open("best-ests/best_est_gbc_{}_scaled.p".format(case[0]), "wb"))
