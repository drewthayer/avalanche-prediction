import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __name__=='__main__':
    avy_df = pd.read_csv('../data/data-clean/avy_data.csv')

    ''' columns:
    ['datetime', 'Obs ID', 'Date', 'Date Known', 'Time', 'Time Known',
       'BC Zone', 'HW Zone', 'Operation', 'Landmark', 'First Name',
       'Last Name', 'HW Path', '#', 'Elev', 'Asp', 'Type', 'Trigger',
       'Trigger_sub', 'Rsize', 'Dsize', 'Incident', 'Area Description',
       'Comments', 'Avg Slope Angle', 'Start Zone Elev',
       'Start Zone Elev units', 'Sliding Sfc', 'Weak Layer', 'Weak Layer Type',
       'Avg Width', 'Max Width', 'Width units', 'Avg Vertical', 'Max Vertical',
       'Vertical units', 'Avg Crown Height', 'Max Crown Height',
       'Crown Height units', 'Terminus ', 'datetime.1'] '''

    # where in Aspen
    aspen_locs = avy_df[avy_df['BC Zone'] == 'Aspen']['Landmark']

    from collections import Counter
    c = Counter(aspen_locs)
    s = sorted(c.items(), key = lambda x: x[1], reverse=True)
    for pair in s:
        plt.bar(pair[0], pair[1])
    plt.xticks( rotation='vertical')
    #plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.tight_layout()
    plt.show()


    # water-year month feature
    month = pd.to_datetime(avy_df.Date).dt.month
    def water_year_month(month):
        if month >= 10:
            month2 = month - 9
        else:
            month2 = month + 3
        return month2
    avy_df['Month_wy'] = month.apply(lambda x: water_year_month(x))

    def water_year_day(day):
        if day >= 273:
            day2 = day - 273
        else:
            day2 = day + 92
        return day2

    dt = pd.to_datetime(avy_df.Date)
    doy = dt.apply(lambda x: x.timetuple().tm_yday)
    avy_df['DOY'] = doy.apply(lambda x: water_year_day(x))

    '''types of avalanches
    ['HS', 'SS', 'WL', 'WS', 'U', 'C', 'L', nan, 'G', 'R', 'SF', 'I'] '''
    # df by avalanche type
    ss = avy_df[avy_df.Type == 'SS']
    wl = avy_df[avy_df.Type == 'WL']
    hs = avy_df[avy_df.Type == 'HS']
    ll = avy_df[avy_df.Type == 'L']
    slab = avy_df[np.in1d(avy_df.Type, ['SS','HS','L'])]

    #plot
    fig, ax = plt.subplots()
    #df_list = [ss, hs, wl, l]
    #names = ['storm slab', 'hard slab', 'wet loose', 'loose']
    df_list = [slab, wl]
    names = ['slab', 'wet loose']
    colors = ['b','g']
    months = ['oct','nov','dec','jan','feb','mar','apr','may','jun','jul','aug','sep']
    d_month = {i:m for i, m in enumerate(months)}

    for idx, df in enumerate(df_list):
        c = Counter(df.Month_wy)
        for k,v in c.items():
            plt.bar(k, v, color=colors[idx], alpha=0.3)
    plt.legend(names)
    plt.xlabel('Water Year Month')
    plt.xticks(range(12), months, rotation='vertical')
    plt.ylabel('# of avalanches')
    plt.title('Monthly distribution of avalanches by type')
    plt.tight_layout()
    plt.show()
    #plt.savefig('../figs/eda/types_by_month.png',dpi=350)
    #plt.close()

    ''' kde of types '''

    # prep data for KDE
    # X_count = np.zeros((len(c), 2))
    # items = [x for x in c.items()]
    # sort = sorted(items, key = lambda x: x[0])
    # for i in range(X.shape[0]):
    #     X_count[i][0] = sort[i][0]
    #     X_count[i][1] = sort[i][1]

    # scipy stats kde
    from scipy.stats import gaussian_kde
    xx = np.linspace(1,12,100)

    fig, ax = plt.subplots()
    for idx, df in enumerate(df_list):
        X = df.Month_wy
        density = gaussian_kde(X)
        density.covariance_factor = lambda : 0.35
        density._compute_covariance()
        plt.plot(xx,density(xx), color=colors[idx])
        plt.hist(X, 30, normed=1, color=colors[idx], alpha=0.5 )
    plt.legend(names)
    plt.xlabel('Water Year Month')
    plt.xticks(range(12), months, rotation='vertical')
    plt.ylabel('# of avalanches (normed)')
    plt.title('Monthly distribution of avalanches by type')
    plt.tight_layout()
    plt.show()

    ''' kde probabilities'''
    feature_cols = [['SS','HS','L'], ['WL']]
    dataframe = avy_df
    target_col = 'DOY'
    xx = np.linspace(1,366,365)
    smooth = 0.35

    def kde_probabilities(dataframe, feature_cols, target_col, xx, smooth):
        probs_list = []
        for features in feature_cols:
            X = dataframe[np.in1d(avy_df.Type, features)][target_col]
            density = gaussian_kde(X)
            density.covariance_factor = lambda : smooth
            density._compute_covariance()
            probs = density(xx)
            probs_list.append(probs)
        return probs_list

    probs_list = kde_probabilities(dataframe, feature_cols, target_col, xx, smooth)
