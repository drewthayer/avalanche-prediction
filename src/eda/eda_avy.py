import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

def water_year_month(month):
    if month >= 10:
        month2 = month - 9
    else:
        month2 = month + 3
    return month2

def water_year_day(day):
    if day >= 273:
        day2 = day - 273
    else:
        day2 = day + 92
    return day2

if __name__=='__main__':
    avy_df = pd.read_csv('../../data/data-clean/avy_data.csv')

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

    # where everywhere

    # plot: where in Aspen
    aspen_locs = avy_df[avy_df['BC Zone'] == 'Aspen']['Landmark']

    from collections import Counter
    c = Counter(aspen_locs)
    s = sorted(c.items(), key = lambda x: x[1], reverse=True)
    for pair in s:
        plt.bar(pair[0], pair[1])
    plt.xticks( rotation='vertical')
    #plt.xticks(range(len(labels)), labels, rotation='vertical')
    #plt.tight_layout()
    plt.show()

    ''' water year month and day '''
    # water-year month feature
    month = pd.to_datetime(avy_df.Date).dt.month
    avy_df['Month_wy'] = month.apply(lambda x: water_year_month(x))

    dt = pd.to_datetime(avy_df.Date)
    doy = dt.apply(lambda x: x.timetuple().tm_yday)
    avy_df['DOY'] = doy.apply(lambda x: water_year_day(x))

    ''' kde probabilities '''
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


    '''types of avalanches
    ['HS', 'SS', 'WL', 'WS', 'U', 'C', 'L', nan, 'G', 'R', 'SF', 'I'] '''
    # df by avalanche type
    ss = avy_df[avy_df.Type == 'SS']
    wl = avy_df[avy_df.Type == 'WL']
    hs = avy_df[avy_df.Type == 'HS']
    ll = avy_df[avy_df.Type == 'L']
    slab = avy_df[np.in1d(avy_df.Type, ['SS','HS','L'])]

    df_list = [slab, wl]
    names = ['slab', 'wet']
    months = ['oct','nov','dec','jan','feb','mar','apr','may','jun','jul','aug','sep']
    d_month = {i:m for i, m in enumerate(months)}

    c_doy_list = []
    for idx, df in enumerate(df_list):
        c = Counter(df.DOY)
        c_doy_list.append(c)

    # figure
    fig, ax1 = plt.subplots()
    # ax 1
    for k,v in c_doy_list[0].items():
        h1 = plt.bar(k, v, color='dodgerblue', alpha=0.2, label='slab', width=5)
    for k,v in c_doy_list[1].items():
        h2 = plt.bar(k, v*4, color='forestgreen', alpha=0.2, label='wet', width=5)
    ax1.set_ylabel('daily # avalanches')
    # ax 2
    ax2 = ax1.twinx()
    h3 = plt.plot(xx, probs_list[0], 'b', linewidth=4)
    h4 = plt.plot(xx, probs_list[1]/4, 'g', linewidth=4)

    ax2.set_ylim([0,0.015])
    ax2.set_ylabel('probability')

    plt.legend(['p(slab)', 'p(wet)'])
    # ax = plt.gca()
    # leg = ax.get_legend()
    # leg.legendHandles[0].set_color('b')
    # leg.legendHandles[1].set_color('g')

    #plt.xlabel('Water Year Month')
    plt.xticks(np.linspace(30,360,12), months, rotation='diagonal')
    #plt.xticks(np.linspace(30,360,12), np.linspace(30,360,12))
    #plt.ylabel('# of avalanches')
    plt.title('seasonal distribution of slab vs. wet avalanches')
    plt.tight_layout()
    plt.show()

    '''just 2 histograms '''
    c_month_list = []
    for idx, df in enumerate(df_list):
        c = Counter(df.Month_wy)
        c_month_list.append(c)
    # figure
    fig, ax1 = plt.subplots()
    # ax 1
    for k,v in c_month_list[0].items():
        h1 = plt.bar(k, v, color='b', alpha=0.2, label='slab')

    ax1.set_ylabel('# avalanches')
    # ax 2
    ax2 = ax1.twinx()
    for k,v in c_month_list[1].items():
        h2 = plt.bar(k, v, color='g', alpha=0.2, label='wet')

    ax2.set_ylim([0,1000])
    ax2.set_ylabel('# of avalanches')

    plt.legend(names)
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('b')
    leg.legendHandles[1].set_color('g')

    plt.xlabel('Water Year Month')
    plt.xticks(range(12), months, rotation='vertical')
    plt.ylabel('# of avalanches')
    plt.title('daily distribution of slab vs. wet avalanches')
    plt.tight_layout()
    plt.show()


    # same plot with histogram
    fig, ax1 = plt.subplots()
    h1 = ax1.hist(slab.Month_wy, color='b', label='slab', alpha=0.5)

    ax2 = ax1.twinx()
    h2 = ax2.hist(wl.Month_wy, color='g', label='wet', alpha=0.5)
    hh = h1 + h2
    h_labels = ['slab','wet']
    plt.legend(hh, h_labels, loc=0)
    plt.show()



    # scipy stats hist, kde
    xx = np.linspace(1,12,100)
    colors = ['b','g']

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



    ''' dsize figure '''
    dd = avy_df[avy_df.Dsize.notnull()]['Dsize']
    dd = dd[dd != 'U']
    # histogram
    c_list = ['y','y','r','r','r','r','r']
    labels, counts = np.unique(dd, return_counts=True)
    plt.bar(labels, counts, align='center', color=c_list, alpha=0.5)
    plt.yticks(np.linspace(0,4000,5))
    #plt.yscale("log", nonposy='clip')
    plt.ylabel('count')
    plt.xlabel('destructive size')
    plt.show()

    ''' frequency ts figure '''
    avy_df['N'] = avy_df['#']
    d2df = avy_df[np.in1d(avy_df.Dsize,['D2','D2.5','D3','D3.5','D4'])]
    aspen2 = d2df[d2df['BC Zone'] == 'Aspen']
    gg = aspen2.groupby('datetime')['N']
    ggs = gg.sum()
    # figure
    fig, ax = plt.subplots(figsize=(18,4))
    plt.bar(pd.to_datetime(ggs.index), ggs, color='teal', width=20, alpha=0.8)
    #plt.plot(pd.to_datetime(ggs.index),ggs)
    plt.title('Aspen zone avalanches, D2 and greater')
    plt.ylabel('Daily # of avalanches')
    plt.ylim([0,27])
    plt.yticks(np.linspace(0,25,6))
    plt.show()
    #plt.savefig('Aspen_navy_ts.png',dpi=350)
    #plt.close()

    ''' frequency avy histogram semi-log '''
    # n_avy histogram
    fig, ax = plt.subplots()
    plt.hist(ggs, 20)
    plt.yscale('log', nonposy='clip')
    plt.xlabel('Daily # of avalanches')
    plt.ylabel('count')
    plt.tight_layout()
    plt.show()

    ''' most frequent backcountry zones '''
    d2df = avy_df[np.in1d(avy_df.Dsize,['D2','D2.5','D3','D3.5','D4'])]
    c_zone = Counter(d2df['BC Zone'])
    zone_keys = sorted(c_zone, key=c_zone.get)
    zone_vals = [c_zone[k] for k in zone_keys]
    alphas = np.linspace(0,1,len(zone_keys))
    fig, ax = plt.subplots()
    for k, v, alpha in zip(zone_keys, zone_vals, alphas):
        plt.barh(k,v, color='b', alpha=alpha)
    plt.xlabel('total # avalanches observed')
    plt.tight_layout()
    plt.show()

    ''' most frequent locations '''
    c_loc = Counter(avy_df.Landmark)
    # list of sorted tuples:
    srt_tupes = sorted(c_loc.items(), key=lambda x: x[1], reverse=True)

    # list of sorted keys
    srt_keys = sorted(c_loc, key=c_loc.get, reverse=True)
    # list of matching sorted values
    vals = [c_loc[x] for x in srt_keys]

    # lat, long for most frequent
    top20 = ['Red Mountain Pass',
             'Marble',
             '10-mile Range',
             'Ophir',
             'Berthoud Pass west side',
             'Ruby Range',
             'Wolf Creek Pass',
             'Cement Creek',
             'Loveland Pass-Southside',
             'Vail Pass',
             'Lizard Head Pass',
             'Mosquito Range',
             'Red Mountain Pass--Eastside',
             'Gore Range',
             'Rocky Mountain National Park',
             'Mount Emmons',
             'Sawatch Range',
             'Richmond Hill',
             'Independence Pass-East side',
             'Ashcroft']
    top_df = avy_df[np.in1d(avy_df.Landmark, top20)]

    # map: most frequent
    import folium



    # old hist plot
    # #plot: histogram of slab vs wet 1
    # fig, ax = plt.subplots()
    # df_list = [slab, wl]
    # names = ['slab', 'wet']
    # colors = ['b','g']
    # months = ['oct','nov','dec','jan','feb','mar','apr','may','jun','jul','aug','sep']
    # d_month = {i:m for i, m in enumerate(months)}
    #
    # for idx, df in enumerate(df_list):
    #     c = Counter(df.Month_wy)
    #     for k,v in c.items():
    #         plt.bar(k, v, color=colors[idx], alpha=0.5)
    # plt.legend(names)
    # plt.xlabel('Water Year Month')
    # plt.xticks(range(12), months, rotation='vertical')
    # plt.ylabel('# of avalanches')
    # plt.title('Monthly distribution of avalanches by type')
    # plt.tight_layout()
    # plt.show()
    #plt.savefig('../figs/eda/types_by_month.png',dpi=350)
    #plt.close()

    # same plot with different scales
    #plot: histogram of slab vs wet 1
