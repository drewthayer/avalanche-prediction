import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import gaussian_kde

from transformation_scripts import water_year_day, water_year_month

def engineer_avy_df(avy_df, bc_zone, min_dsize=2):
    # D scale to ordinal
    tmp = avy_df['Dsize'].fillna("D0")
    tmp = tmp.apply(lambda x: "D0" if x == "U" else x )
    avy_df['D'] = tmp.apply(lambda x: float(x.split("D")[1]))

    # could do this with ordinal "D" now
    avy_df['N_AVY'] = np.where(avy_df.D >= min_dsize, avy_df['#'], 0)

    # new dataframe and groupby object
    zone_df = pd.DataFrame()
    subset = avy_df[avy_df['BC Zone'] == bc_zone].groupby('datetime')
    # n avalanches
    zone_df['N_AVY'] = subset['N_AVY'].sum()
    zone_df['dt'] = pd.to_datetime(zone_df.index)
    # water year month
    month = zone_df['dt'].dt.month
    zone_df['MONTH'] = month.apply(lambda x: water_year_month(x))
    # day of water year
    doy = zone_df['dt'].apply(lambda x: x.timetuple().tm_yday)
    zone_df['DOY'] = doy.apply(lambda x: water_year_day(x))
    # n_avy in last 24 hours
    temp = zone_df['N_AVY'].values
    temp = np.insert(temp,0,0)
    temp = np.delete(temp, -1)
    zone_df['N_AVY_24'] = temp
    # n_avy D sum 24
    zone_df['D_SUM'] = subset['D'].sum() # includes D1
    temp = zone_df['D_SUM'].values
    temp = np.insert(temp,0,0)
    temp = np.delete(temp, -1)
    zone_df['DSUM_24'] = temp
    # kde probabilities of slab/wet
    probs = pickle.load( open( 'pkl/kde_probs.p', 'rb'))
    p_slab = probs[0]
    p_wet = probs[1]
    zone_df['P_SLAB'] = zone_df.DOY.apply(lambda x: p_slab[x])
    zone_df['P_WET'] = zone_df.DOY.apply(lambda x: p_wet[x])
    # features: slab, wet
    zone_df['Type'] = subset.Type
    tmp = zone_df.Type
    wet = tmp.apply(lambda x: 1 if 'WL' in set(x[1]) else 0)
    slab = tmp.apply(lambda x: 1 if ('HS' in set(x[1])) or ('SS' in set(x[1])) else 0)
    zone_df['WET'] = wet
    zone_df['SLAB'] = slab

    # set index to timstamp object
    zone_df.set_index(zone_df['dt'], inplace=True)
    zone_df.drop('dt', axis=1, inplace=True)

    cols_to_keep = ['N_AVY','MONTH','DOY','N_AVY_24','DSUM_24','P_SLAB','P_WET',
                    'WET', 'SLAB']
    out = zone_df[cols_to_keep]

    return out

def engineer_wind_df(airport_df, airport_list):
    airport_df['dt'] = pd.to_datetime(airport_df['datetime'])
    airport_df.set_index(airport_df['dt'], inplace=True)
    airport_df.drop('datetime', axis=1, inplace=True)


    wind_df = pd.DataFrame()

    for name in airport_list:
        df = airport_df[airport_df['airport'] == name]
        wind_df['WSP_SSTN_{}'.format(name)] = df['SustainedWindSpeed']
        wind_df['WSP_PEAK_{}'.format(name)] = df['Daily_peak_wind']

    return wind_df

def engineer_timelag_features(df, col_list, lag=3):
    lags = range(lag)
    for colname in col_list:
        col = df[colname].values
        for i in lags:
            col_new = np.insert(col,0,0) # add a zero in first element
            col_new = np.delete(col_new,-1) # remove last element
            df[colname+'_{}'.format(i+1)] = col_new
            col = col_new

    return df

def engineer_snotel_df(snotel_df, start_date):
    snotel_df['datetime'] = pd.to_datetime(snotel_df['dt'])
    snotel_df = snotel_df[snotel_df['datetime'] >= start_date]
    # amount
    snotel_df['DEPTH'] = snotel_df['precip_start_m']
    # amount > 60 cm
    snotel_df['GRTR_60'] = np.where(snotel_df.precip_start_m > 0.6, snotel_df.precip_start_m - 0.6, 0)
    # snow in last 24 hrs
    snotel_df['SNOW_24'] = snotel_df['precip_incr_m']
    # Weighted sum of snow fall in last 4 days: weights = (1.0, 0.75, 0.50, 0.25)
    snotel_df = engineer_timelag_features(snotel_df, ['SNOW_24'], lag=4)
    snotel_df['SNOW_4DAY'] = snotel_df.SNOW_24_1 + snotel_df.SNOW_24_2*0.75 + snotel_df.SNOW_24_3*0.5 + snotel_df.SNOW_24_4*0.25
    # water content of new snow
    temp = snotel_df['swe_start_m'].values
    temp = np.insert(temp,0,0)
    temp = np.delete(temp, -1)
    snotel_df['SWE_24'] = snotel_df['swe_start_m'] - temp
    # Density of new snow, ratio of water content of new snow to new snow depth
    snotel_df['DENSE_24'] = snotel_df['SWE_24'] / snotel_df['SNOW_24'] # need to remove nan, inf

    # Change in TOTSTK60 relative to depth of snowfall in the last 24 hours
    temp = snotel_df['GRTR_60'].values
    temp = np.insert(temp,0,0)
    temp = np.delete(temp, -1)
    snotel_df['SETTLE'] = (snotel_df['GRTR_60'] - temp) / snotel_df['SNOW_24'] # nans, inf

    # min temp last night
    snotel_df['TMIN'] = snotel_df['airtemp_min_C']
    # Difference in minimum temperature from previous day
    temp = snotel_df['TMIN'].values
    temp = np.insert(temp,0,0)
    temp = np.delete(temp, -1)
    snotel_df['TMIN_DELTA'] = snotel_df['TMIN'] - temp
    # max temp in last 2 hours
    snotel_df['TMAX'] = snotel_df['airtemp_max_C']
    # sum of max temp of last 4 days
    snotel_df = engineer_timelag_features(snotel_df, ['TMAX'], lag=4)
    snotel_df['TMAX_SUM'] = snotel_df.TMAX_1 + snotel_df.TMAX_2 + snotel_df.TMAX_3 + snotel_df.TMAX_4
    # convert station to just number
    snotel_df['STATION'] = snotel_df.station.str.split('_').apply(lambda x: x[0])
    # columns to keep
    cols = ['datetime', 'DEPTH', 'GRTR_60', 'SNOW_24', 'SNOW_4DAY','SWE_24',
            'DENSE_24','SETTLE','TMIN','TMIN_DELTA','TMAX','TMAX_SUM','STATION']
    snow_df = snotel_df[cols]
    # set index to datetime
    snow_df.set_index('datetime', inplace=True)
    # replace inf with nan
    snow_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return snow_df



if __name__=='__main__':
    # paths
    current = os.getcwd()
    clean_dir = ''.join([current,'/../data/data-clean/'])
    # load data
    avy_df = pd.read_csv(clean_dir + 'avy_data.csv')
    airport_df = pd.read_csv(clean_dir + 'airport_data.csv')
    snotel_df = pd.read_csv(clean_dir + 'snotel_data.csv')

    # engineer CAIC data
    zone_df = engineer_avy_df(avy_df, 'Aspen', min_dsize=2)

    # time range:
    start_date = zone_df.index.min()
    end_date = zone_df.index.max()

    # engineer wind data
    airport_list = ['aspen', 'leadville']
    wind_df = engineer_wind_df(airport_df, airport_list)

    # engineer snotel data
    snow_df = engineer_snotel_df(snotel_df, start_date)
    pickle.dump( snow_df, open( "pkl/snow_df.p", "wb" ) )


    # remove 2018 data
    #snow_df.drop(snow_df[snow_df.year == 2018].index, inplace=True)

    # remove rows with swe=0
    #snow_df.drop(snow_df[snow_df.swe_start_m == 0].index, inplace=True)

    # remove september data (highly volatile and uncharacteristic)
    #snow_df.drop(snow_df[snow_df.month == 9].index, inplace=True)
    stationnames = ['618_mcclure_pass',
                    '669_north_lost_trail',
                    '737_schofield_pass',
                    '542_independence_pass',
                    '369_brumley',
                    '547_ivanhoe']

    # to test, use only one station
    snow_df = snow_df[snow_df.STATION == '542']

    ''' assemble feature matrix '''
    merge = pd.merge(zone_df, wind_df, how='left', left_index=True, right_index=True)
    merge_all = pd.merge(merge, snow_df, how='left', left_index=True, right_index=True)

    # remove non-numeric columns
    merge_all.drop('STATION', axis=1, inplace=True)

    # test for nans
    test_nans = np.sum(merge_all.isna())

    ''' save to pickle '''
    pickle.dump( merge_all, open( "pkl/aspen_d2_slabwet_labeled.p", "wb" ) )

    # diagnostics
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].plot(merge_all.P_SLAB, merge_all.SLAB, 'ok')
    ax[0].set_title('slab avalanche')
    ax[0].set_xlabel('p(slab)')
    ax[0].set_ylabel('events')
    ax[1].plot(merge_all.P_WET, merge_all.WET, 'ok')
    ax[1].set_title('wet avalanche')
    ax[1].set_xlabel('p(wet)')
    plt.tight_layout()
    plt.show()
