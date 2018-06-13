import pickle
import numpy as np
import pandas as pd
import os
import sys
# my scripts
from transformation_scripts import water_year_day, water_year_month
from sqlite3_scripts import connect_to_sql, create_table_sql, write_pandas_to_sql

def engineer_avy_df(avy_df, bc_zone, min_dsize=2):
    ''' feature engineering for avalanche dataframe
    input: pandas df
    arguments: backcountry zone (string), minumum dsize(int)
    output: pandas df
    '''
    # convert D scale to ordinal
    tmp = avy_df['Dsize'].fillna("D0")
    tmp = tmp.apply(lambda x: "D0" if x == "U" else x )
    avy_df['D'] = tmp.apply(lambda x: float(x.split("D")[1]))

    # select avalanches > min_dsize
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
    out_df = zone_df[cols_to_keep]

    return out_df

def engineer_wind_df(airport_df, airport_list):
    ''' feature engineering for wind speed dataframe
    input: pandas df
    output: pandas df
    '''
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
    ''' creates new features from previous days for "lag"=n days
        features in "col_list"

        input: pandas df
        output: pandas df
    '''
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
    ''' feature engineering for snotel dataframe
    input: pandas df
    output: pandas df
    '''
    snotel_df['datetime'] = pd.to_datetime(snotel_df['dt'])
    snotel_df = snotel_df[snotel_df['datetime'] >= start_date]
    # amount
    snotel_df['DEPTH'] = snotel_df['precip_start_m']
    # amount > 60 cm
    snotel_df['GRTR_40'] = np.where(snotel_df.precip_start_m > 0.4, snotel_df.precip_start_m - 0.4, 0)
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
    temp = snotel_df['GRTR_40'].values
    temp = np.insert(temp,0,0)
    temp = np.delete(temp, -1)
    snotel_df['SETTLE'] = (snotel_df['GRTR_40'] - temp) / snotel_df['SNOW_24'] # nans, inf

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
    cols = ['datetime', 'DEPTH', 'GRTR_40', 'SNOW_24', 'SNOW_4DAY','SWE_24',
            'DENSE_24','SETTLE','TMIN','TMIN_DELTA','TMAX','TMAX_SUM','STATION']
    snow_df = snotel_df[cols]
    # set index to datetime
    snow_df.set_index('datetime', inplace=True)
    # replace inf with nan
    snow_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return snow_df

def df_simple_impute(df, method='mean'):
    ''' impute missing values for numeric coluns in a pandas df
    options: inpute with mean or zero (input as argument)
    input: pandas df
    output: pandas df
    '''
    # define simple imputer
    def simple_impute(x, val):
        if np.isnan(x):
            return val
        else:
            return x
    # apply to numeric columns
    for col in df.columns:
        if df[col].dtype == 'float64':
            column = df[col]
            colmean = column.mean()
            if method == 'mean':
                val = colmean
            elif method == 'zero':
                val = 0
            df[col] = column.apply(lambda x: simple_impute(x, val))
    return df

if __name__=='__main__':
    ''' set name variable from command line
        names can be 'aspen' or 'nsj' '''

    zonename = sys.argv[1] # 'aspen' or 'nsj'
    # labels, ids corresponding with each zone
    zone_labels = {'aspen': 'Aspen', 'nsj':'Northern San Juan'}
    zone_stationid = {'aspen':'542', 'nsj':'713'}
    # paths
    current = os.getcwd()

    # load avalanche data from sql
    conn = connect_to_sql(current + '/../data/data-caic.db')
    avy_df = pd.read_sql("select * from avalanche", conn)

    # load zone snow, weather data from sql
    conn = connect_to_sql(current + '/../data/data-{}.db'.format(zonename))
    airport_df = pd.read_sql("select * from airport", conn)
    snotel_df = pd.read_sql("select * from snotel", conn)
    conn.close()

    # impute missing values
    avy_imputed = df_simple_impute(avy_df, method='mean')
    airport_imputed = df_simple_impute(airport_df, method='mean')
    snotel_imputed = df_simple_impute(snotel_df, method='mean')

    # engineer CAIC data for specific zone
    zone_df = engineer_avy_df(avy_imputed, zone_labels[zonename], min_dsize=2)

    # select time range:
    start_date = zone_df.index.min()
    end_date = zone_df.index.max()

    # engineer wind data
    airport_list = [x for x in airport_df.airport.unique()]
    wind_df = engineer_wind_df(airport_imputed, airport_list)

    # engineer snotel data
    snow_df = engineer_snotel_df(snotel_imputed, start_date)
    snow_df = df_simple_impute(snow_df, method='zero') # impute zero for engineered features

    # to test, use only most representative station
    snow_df = snow_df[snow_df.STATION == zone_stationid[zonename]]

    #assemble feature matrix
    merge = pd.merge(snow_df, wind_df, how='left', left_index=True, right_index=True)
    merge_all = pd.merge(merge, zone_df, how='left', left_index=True, right_index=True)

    # remove non-numeric columns, impute remaining nans with zero
    merge_all.drop('STATION', axis=1, inplace=True)
    merge_imputed = df_simple_impute(merge_all, method='zero')

    #save to sql db
    db = current + '/../data/data-engineered.db'
    tablename = zonename
    conn = connect_to_sql(db)
    write_pandas_to_sql(conn, tablename, merge_imputed)
    conn.close()
