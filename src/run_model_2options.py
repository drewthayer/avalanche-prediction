from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
# import transformations
from transformation_scripts import oversample
from plotting_scripts import feat_importance_plot


if __name__=='__main__':
    # load data
    df = pickle.load( open( 'pkl/aspen_d2_slabwet_labeled.p', 'rb'))
    # the dreaded fill with 0
    df.fillna(0, inplace=True)

    ''' N_AVY when case = slab/wet '''
    cases = [['SLAB','WET'], ['WET','SLAB']]
    n_oversamps = [15, 6]
    c_true = ['b','g']
    c_pred = ['r','orange']
    preds = []
    n_avy = df.N_AVY
    # drop related columns
    df.drop('N_AVY', axis=1, inplace=True)

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    for idx, case in enumerate(cases):
        print('case: {}'.format(case[0]))
        data_df = df.copy()
        # drop other binary column
        data_df.drop(case[1], axis=1, inplace=True)
        # create target column
        data_df['Target'] = n_avy * data_df[case[0]]
        ycol = 'Target'

        # train test split in time
        splitdate = pd.to_datetime('2016-06-01')
        train_df = data_df[data_df.index <= splitdate]
        test_df = data_df[data_df.index > splitdate]

        # oversample train data
        n = n_oversamps[idx]
        train_shuffle, counts, factors = oversample(train_df, 'Target', n=n)
        print('oversample to n = {}'.format(n))
        #pickle.dump( oversamp_df, open( "pkl/aspen_oversamp6.p", "wb" ) )
        #train_shuffle = train_df

        ''' select features and target X,y '''
        # define target and remove target nans

        # train set
        X_train = train_shuffle.copy()
        y_train = X_train.pop(ycol)
        # test set
        X_test = test_df.copy()
        y_test = X_test.pop(ycol)
        # datetime for plot
        test_datetime = pd.to_datetime(X_test.index)

        ''' random forest '''
        best_params = {'criterion': 'mse',
            'max_depth': 14,
            'max_features': 'auto',
            'min_samples_leaf': 2,
            'min_samples_split': 4,
            'n_estimators': 500,
            'n_jobs': -1,
            'oob_score':True}
        rfr = RandomForestRegressor(**best_params)
        rfr.fit(X_train, y_train)
        # metrics
        oob = rfr.oob_score_
        print('rfr out-of-bag train score = {:0.3f}'.format(oob))
        importances_rfr = rfr.feature_importances_
        rfr_feats = sorted(zip(X_train.columns, importances_rfr), key=lambda x:abs(x[1]), reverse=True)
        # predictions
        preds_rfr = rfr.predict(X_test)
        preds.append(preds_rfr)

        rmse = np.sqrt(np.sum((y_test - preds_rfr)**2)/len(y_test))
        print('rfr test rmse = {:0.3f}'.format(rmse))
        #score = accuracy_score(y_test, preds_rfr)
        #print('rfr test accuracy = {:0.3f}'.format(score))

        # plot
        h1 = ax.plot(test_datetime,y_test,c_true[idx],
                    label='actual {}'.format(case[0]))
        h2 = ax.plot(test_datetime,preds_rfr,c_pred[idx],
                    linestyle = 'dashed',
                    label='predicted {}'.format(case[0]))
        ax.set_ylabel('daily # of avalanches')
        ax.set_title('Aspen, CO: avalanches >= D2')
    plt.legend()
    #plt.xticks(test_datetime, rotation='vertical')
    plt.show()
    #plt.savefig('../figs/rfr_d2_slabwet.png', dpi=250)
    #plt.close()

    # feature importance plot
    names = X_train.columns
    filename = '../figs/rfr_d2_slabwet_feats.png'
    #feat_importance_plot(rfr,names,filename,color='g',alpha=0.5,fig_size=(10,10),dpi=250)

    # ''' test: simple model with only DSUM '''
    # # define target and remove target nans
    # ycol = 'N_AVY'
    # xcol = 'D_SUM'
    # # train set
    # X_train = train_shuffle.copy()
    # y_train = X_train.pop(ycol)
    # X_train = X_train[xcol]
    # # test set
    # X_test = test_df.copy()
    # y_test = X_test.pop(ycol)
    # X_test = X_test[xcol]
