from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pdb

from scripts.modeling_scripts import train_test_split_time, predict_classifier,
        print_scores


def multi_case_partial_dependence(df, cases, ests, stdzrs,
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
        names = list(X_train.columns)
        features = [11, 12, 13, 14, (9, 18)]
        # plot
        fig, axs = plot_partial_dependence(est, X_train, features,
                                           feature_names=names,
                                           n_jobs=3, grid_resolution=50)
        fig.suptitle('Partial dependence of features\n'
                     'for {} model'.format(case[0]))
        plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

        print('Custom 3d plot via ``partial_dependence``')
        fig = plt.figure()

        target_feature = (9, 18)
        pdp, axes = partial_dependence(est, target_feature,
                                       X=X_train, grid_resolution=50)
        XX, YY = np.meshgrid(axes[0], axes[1])
        Z = pdp[0].reshape(list(map(np.size, axes))).T
        ax = Axes3D(fig)
        surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                               cmap=plt.cm.BuPu, edgecolor='k')
        ax.set_xlabel(names[target_feature[0]])
        ax.set_ylabel(names[target_feature[1]])
        ax.set_zlabel('Partial dependence')
        #  pretty init view
        ax.view_init(elev=22, azim=122)
        plt.colorbar(surf)
        plt.suptitle('Partial dependence of features\n'
                     'for {} model'.format(case[0]))
        plt.subplots_adjust(top=0.9)

        plt.show()




    #return y_true_l, y_hat_l, y_proba_l, feats_l, list(y_test.index)


if __name__=='__main__':
    # load data
    df = pickle.load( open( 'pkl/aspen_d2_imputemean_alldays.p', 'rb'))
    df.drop('N_AVY', axis=1, inplace=True)
    #df.drop('MONTH', axis=1, inplace=True)
    # fill na with zero in case any not imputed
    df.fillna(0, inplace=True)

    #load fitted models
    est_slab, std_slab = pickle.load( open( 'best-ests/best_est_gbc_SLAB_scaled.p', 'rb'))
    est_wet, std_wet = pickle.load( open( 'best-ests/best_est_gbc_WET_scaled.p', 'rb'))

    # load fitted standardizers

    ''' N_AVY when case = slab/wet '''
    params = {
    'cases': [['SLAB','WET'], ['WET','SLAB']],
    'ests': [est_slab, est_wet],
    'stdzrs': [std_slab, std_wet],
    'c_true': ['b','g'],
    'c_pred': ['r','orange'],
    'n_oversamps': [1,1]}

    outputs = multi_case_partial_dependence(df, **params)
