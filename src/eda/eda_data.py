import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

if __name__=='__main__':
    # read data
    data_df = pickle.load( open('pkl/aspen_1.p', 'rb'))

    y = data_df.pop('N_AVY')
    X = data_df
    X.fillna(0, inplace=True)

    ''' columns:
    ['MONTH', 'DOY', 'N_AVY_24', 'DSUM_24', 'WSP_SSTN_aspen',
       'WSP_PEAK_aspen', 'WSP_SSTN_leadville', 'WSP_PEAK_leadville', 'DEPTH',
       'GRTR_60', 'SNOW_24', 'SNOW_4DAY', 'SWE_24', 'DENSE_24', 'SETTLE',
       'TMIN', 'TMIN_DELTA', 'TMAX', 'TMAX_SUM']
       '''

    # cluster
    pipe = Pipeline([
        ('standardize', StandardScaler()),
        ('cluster', KMeans(n_clusters=3))
        ])

    pipe.fit(X)
    centers = pipe.named_steps['cluster'].cluster_centers_
    y_hat = pipe.predict(X)

    xx = range(len(X.columns))
    for c in centers:
        plt.plot(xx, c, marker='o')
    plt.show()

    # cluster plot
    fig, ax = plt.subplots()
    im = ax.scatter(X['MONTH'],X['WSP_SSTN_aspen'], c = y_hat)
    cax = np.linspace(0,max(y_hat),3)
    plt.colorbar(im)
    plt.show()




    # # pca
    # pipe = Pipeline([
    #     ('standardize', StandardScaler()),
    #     ('dim_reduce', PCA(n_components=2))
    #     ])
    #
    # pipe.fit(X)
    # Xt = pipe.fit_transform(X)
    # comps = pipe.named_steps['dim_reduce'].components_
    # expl = pipe.named_steps['dim_reduce'].explained_variance_
    #
    # xx = range(len(X.columns))
    # for c in comps:
    #     plt.plot(xx, c, marker='o')
    # plt.show()
    #
    # # PCA component plot
    # fig, ax = plt.subplots()
    # im = ax.scatter(Xt[:,0],Xt[:,1], c = X['DSUM_24'])
    # cax = [0,1,10,100]
    # plt.colorbar(im)
    # plt.show()
