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
from plotting_scripts import feat_importance_plot, output_histograms

def run_classifier(df, cases, n_oversamps, c_true, c_pred, model):
    y_true = []
    preds = []
    probas = []
    n_avy = df.N_AVY

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    for idx, case in enumerate(cases):
        print('case: {}'.format(case[0]))
        data_df = df.copy()
        # drop other binary column
        data_df.drop(case[1], axis=1, inplace=True)
        # create target column
        data_df['AVY'] = np.where(df.N_AVY == 0, 1, 0)
        ycol = 'AVY'
        data_df.drop('N_AVY', axis=1, inplace=True)
        # drop target binary column
        data_df.drop(case[0], axis=1, inplace=True)

        # train test split in time
        splitdate = pd.to_datetime('2016-06-01')
        train_df = data_df[data_df.index <= splitdate]
        test_df = data_df[data_df.index > splitdate]

        # oversample train data
        #n = n_oversamps[idx]
        #train_shuffle, counts, factors = oversample(train_df, 'AVY', n=n)
        #print('oversample to n = {}'.format(n))
        #pickle.dump( oversamp_df, open( "pkl/aspen_oversamp6.p", "wb" ) )
        train_shuffle = train_df

        ''' select features and target X,y '''
        # train set
        X_train = train_shuffle.copy()
        y_train = X_train.pop(ycol)
        # test set
        X_test = test_df.copy()
        y_test = X_test.pop(ycol)
        # datetime for plot
        test_datetime = pd.to_datetime(X_test.index)

        ''' run model '''
        model.fit(X_train, y_train)
        # metrics
        #oob = model.oob_score_
        #print('out-of-bag train score = {:0.3f}'.format(oob))
        importances_rfr = model.feature_importances_
        rfr_feats = sorted(zip(X_train.columns, importances_rfr), key=lambda x:abs(x[1]), reverse=True)
        # predictions
        y_hat = model.predict(X_test)
        proba = model.predict_proba(X_test)
        # save true, predicted for return
        preds.append(y_hat)
        probas.append(proba)
        y_true.append(y_test)


        #rmse = np.sqrt(np.sum((y_test - preds)**2)/len(y_test))
        #print('test rmse = {:0.3f}'.format(rmse))
        score = accuracy_score(y_test, y_hat)
        print('rfc test accuracy = {:0.3f}'.format(score))

        # feature importance plot
        names = X_train.columns
        filename = '../figs/rfr_d2_2_class{}.png'.format(case[0])
        feat_importance_plot(model,names,filename,color='g',
            alpha=0.5,fig_size=(10,10),dpi=250)

        # plot
        h1 = ax.plot(test_datetime,y_test,c_true[idx],
                    label='actual {}'.format(case[0]))
        h2 = ax.plot(test_datetime,y_hat,c_pred[idx],
                    linestyle = 'dashed',
                    label='predicted {}'.format(case[0]))
        ax.set_ylabel('daily # of avalanches')
        ax.set_title('Aspen, CO: avalanches >= D2')
    plt.legend()
    #plt.xticks(test_datetime, rotation='vertical')
    plt.show()
    #plt.savefig('../figs/rfr_d2_slabwet.png', dpi=250)
    #plt.close()

    return y_true, preds, probas

if __name__=='__main__':
    # load data
    df = pickle.load( open( 'pkl/aspen_d2_imputemean.p', 'rb'))
    # fill na with zero in case any not imputed
    df.fillna(0, inplace=True)

    ''' N_AVY when case = slab/wet '''
    cases = [['SLAB','WET'], ['WET','SLAB']]
    n_oversamps = [15, 6]
    c_true = ['b','g']
    c_pred = ['r','orange']

    best_params = {'criterion': 'mse',
        'max_depth': 14,
        'max_features': 'auto',
        'min_samples_leaf': 2,
        'min_samples_split': 4,
        'n_estimators': 500,
        'n_jobs': -1,
        'oob_score':True}
    #rfr = RandomForestRegressor(**best_params)
    rfc = RandomForestClassifier()

    y_true, preds, proba = run_classifier(df, cases, n_oversamps, c_true, c_pred, model=rfc)

    output_histograms(y_true, preds)

    # save outputs to pkl
    pickle.dump((y_true,preds), open('pkl/aspen_d2_class.p','wb'))
