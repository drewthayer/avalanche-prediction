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

from plotting_scripts import feat_importance_plot

def predict_classifier(X_test, y_test, model):
    ''' applies .predict() method of fitted classifier to X,y data '''
    y_hat = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    importances = model.feature_importances_
    return y_hat, y_proba, importances

def print_scores(y_true, y_hat, method_list):
    for method in method_list:
        score = method(y_true, y_hat)
        print('test {} = {:0.3f}'.format(method.__name__, score))

def multi_case_classifier_predict(df, cases, ests,
                n_oversamps, c_true, c_pred):
    y_true_l = []
    y_hat_l = []
    y_proba_l = []
    feats_l = []

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    for case, est, n, c_t, c_p in zip(cases, ests, n_oversamps, c_true, c_pred):
        data_df = df.copy() # copy to read all columns after dropping
        print('case: {}'.format(case[0]))

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
        # datetime for plot
        test_datetime = pd.to_datetime(X_test.index)


        ''' predict with fitted model  '''
        y_hat, y_proba, importances = predict_classifier(X_test, y_test, est)
        feats = sorted(zip(X_train.columns, importances), key=lambda x:abs(x[1]), reverse=True)

        # save true, predicted, proba, feats
        y_true_l.append(y_test)
        y_hat_l.append(y_hat)
        y_proba_l.append(y_proba)
        feats_l.append(feats)

        # print scores
        method_list = [accuracy_score, recall_score, precision_score]
        print_scores(y_test, y_hat, method_list)

        # feature importance plot
        # names = X_train.columns
        # filename = '../figs/rfr_d2_2_class{}.png'.format(case[0])
        # feat_importance_plot(model,names,filename,color='g',
        #     alpha=0.5,fig_size=(10,10),dpi=250)

        # plot
        h1 = ax.plot(test_datetime,y_test,c_t,
                    label='actual {}'.format(case[0]))
        h2 = ax.plot(test_datetime,y_hat,c_p,
                    linestyle = 'dashed',
                    label='predicted {}'.format(case[0]))
        ax.set_ylabel('daily # of avalanches')
        ax.set_title('Aspen, CO: avalanches >= D2')
    plt.legend()
    #plt.xticks(test_datetime, rotation='vertical')
    plt.show()
    #plt.savefig('../figs/rfr_d2_slabwet.png', dpi=250)
    #plt.close()

    return y_true_l, y_hat_l, y_proba_l, feats_l, list(y_test.index)


if __name__=='__main__':
    # load data
    df = pickle.load( open( 'pkl/aspen_d2_imputemean_alldays.p', 'rb'))
    df.drop('N_AVY', axis=1, inplace=True)
    # fill na with zero in case any not imputed
    df.fillna(0, inplace=True)

    #load fitted models
    est_slab = pickle.load( open( 'best-ests/best_est_rfc_slab.p', 'rb'))
    est_wet = pickle.load( open( 'best-ests/best_est_rfc_wet.p', 'rb'))

    ''' N_AVY when case = slab/wet '''
    params = {
    'cases': [['SLAB','WET'], ['WET','SLAB']],
    'ests': [est_slab, est_wet],
    'c_true': ['b','g'],
    'c_pred': ['r','orange'],
    'n_oversamps': [1,1]}

    y_true_l, y_hat_l, y_proba_l, feats_l, test_ts = multi_case_classifier_predict(df, **params)

    # save outputs to pkl
    pickle.dump((y_true_l, y_hat_l, y_proba_l, feats_l, test_ts),
            open('pkl/aspen_d2_rfc_best_output.p','wb'))
