from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
# import transformations
from transformation_scripts import oversample
from plotting_scripts import feat_importance_plot


if __name__=='__main__':
    # load data
    data_df = pickle.load( open( 'pkl/aspen_d2_slabwet.p', 'rb'))
    # the dreaded fill with 0
    data_df.fillna(0, inplace=True)

    # train test split in time
    splitdate = pd.to_datetime('2016-06-01')
    train_df = data_df[data_df.index <= splitdate]
    test_df = data_df[data_df.index > splitdate]

    # oversample train data
    train_shuffle, counts, factors = oversample(train_df, 'N_AVY', n=15)
    #pickle.dump( oversamp_df, open( "pkl/aspen_oversamp6.p", "wb" ) )
    #train_shuffle = train_df

    ''' select features and target X,y '''
    # define target and remove target nans
    ycol = 'N_AVY'
    # train set
    X_train = train_shuffle.copy()
    y_train = X_train.pop(ycol)
    # test set
    X_test = test_df.copy()
    y_test = X_test.pop(ycol)

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
    rmse = np.sqrt(np.sum((y_test - preds_rfr)**2)/len(y_test))
    print('rfr test rmse = {:0.3f}'.format(rmse))

    # plot
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    h1 = ax.plot(range(len(y_test)),y_test,'b', label='actual')
    h2 = ax.plot(range(len(preds_rfr)),preds_rfr,'orange', label='predicted - rfr')
    ax.set_ylabel('daily # of avalanches')
    ax.set_title('Aspen, CO: avalanches >= D2')
    ax.legend()
    plt.savefig('../figs/rfr_d2_slabwet.png', dpi=250)
    plt.close()

    # feature importance plot
    names = X_train.columns
    filename = '../figs/rfr_d2_slabwet_feats.png'
    feat_importance_plot(rfr,names,filename,color='g',alpha=0.5,fig_size=(10,10),dpi=250)

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
