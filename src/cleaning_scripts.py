import numpy as np
import pandas as pd

def remove_airtemp_outliers(df):
    '''
    input: pandas dataframe
        - snotel data with
        - columns re-named for degrees C

    output: pandas dataframe
    '''
    import pandas as pd
    # clear min airtemp outliers
    df.drop(df[df.airtemp_min_C > 18].index, inplace=True)
    df.drop(df[df.airtemp_min_C < -40].index, inplace=True)
    # clear max airtemp outliers
    df.drop(df[df.airtemp_max_C > 50].index, inplace=True)
    df.drop(df[df.airtemp_max_C < -45].index, inplace=True)

    return df

def clean_snow_data(dataframe):
    '''
    input: pandas dataframe
        - snotel data
        - header removed
        - columns named

    output: pandas dataframe
        - index set to dt
    '''
    snow_df = dataframe
    # unit conversions to metric
    snow_df['swe_start_m'] = snow_df.swe_start_in * 0.0254
    snow_df['airtemp_max_C'] = 5/9*(snow_df.airtemp_max_F - 32)
    snow_df['airtemp_min_C'] = 5/9*(snow_df.airtemp_min_F - 32)
    snow_df['airtemp_mean_C'] = 5/9*(snow_df.airtemp_mean_F - 32)
    snow_df['precip_start_m'] = snow_df.precip_start_in * 0.0254
    snow_df['precip_incr_m'] = snow_df.precip_incr_in * 0.0254

    # drop standard unit columns
    snow_df.drop(['swe_start_in'], axis=1, inplace=True)
    snow_df.drop(['airtemp_max_F'], axis=1, inplace=True)
    snow_df.drop(['airtemp_min_F'], axis=1, inplace=True)
    snow_df.drop(['airtemp_mean_F'], axis=1, inplace=True)
    snow_df.drop(['precip_start_in'], axis=1, inplace=True)
    snow_df.drop(['precip_incr_in'], axis=1, inplace=True)

    # datetime operations
    snow_df['dt'] = pd.to_datetime(snow_df['dt_string'])
    snow_df['year'] = snow_df['dt'].dt.year
    snow_df['month'] = snow_df['dt'].dt.month

    # drop datetime string column
    snow_df.drop(['dt_string'], axis=1, inplace=True)

    #set snow df index to dt
    snow_df.set_index(snow_df.dt, inplace=True)

    return snow_df

def clean_airport_data(df,name):
    airport_df = df
    # columns to select
    cols = ['STATION','ELEVATION','DATE','DAILYDeptFromNormalAverageTemp',
       'DAILYAverageRelativeHumidity', 'DAILYAverageDewPointTemp',
       'DAILYAverageWetBulbTemp',
       'DAILYPrecip', 'DAILYSnowfall', 'DAILYSnowDepth',
       'DAILYAverageStationPressure', 'DAILYAverageSeaLevelPressure',
       'DAILYAverageWindSpeed', 'DAILYPeakWindSpeed', 'PeakWindDirection',
       'DAILYSustainedWindSpeed', 'DAILYSustainedWindDirection']
    airport_df = airport_df[cols]

    # get only data with a DAILYAverageWindSpeed feature
    airport_df = airport_df[~np.isnan(airport_df['DAILYAverageWindSpeed'])]
    airport_df['day'] = airport_df.DATE.str.rsplit(' ',n=1).str[0]
    airport_df['datetime'] = pd.to_datetime(airport_df.day)
    airport_df.set_index(airport_df.datetime, inplace=True)

    # clean up columns the easy way (mixed types, some loss) DAILYSustainedWindSpeed, DAILYSustainedWindDirection
    airport_df['SustainedWindDirection'] = airport_df['DAILYSustainedWindDirection'].convert_objects(convert_numeric=True)
    airport_df['SustainedWindSpeed'] = airport_df['DAILYSustainedWindSpeed'].convert_objects(convert_numeric=True)
    airport_df['DeptFromNormalAvgTemp'] = airport_df['DAILYDeptFromNormalAverageTemp'].convert_objects(convert_numeric=True)
    airport_df['Precip'] = airport_df['DAILYPrecip'].convert_objects(convert_numeric=True)
    airport_df['Daily_peak_wind'] = airport_df['DAILYPeakWindSpeed'].convert_objects(convert_numeric=True)
    airport_df['Peak_wind_direction'] = airport_df['PeakWindDirection'].convert_objects(convert_numeric=True)

    # add 'name' to col tags
    save_cols = ['DeptFromNormalAvgTemp', 'DAILYAverageRelativeHumidity',
    'DAILYAverageDewPointTemp', 'DAILYAverageWetBulbTemp','DAILYAverageWindSpeed',
    'Daily_peak_wind', 'Peak_wind_direction', 'SustainedWindSpeed',
    'SustainedWindDirection']

    # subset of columns to save
    save_cols_less = ['Daily_peak_wind', 'SustainedWindSpeed']

    # for multiple labeled columns:
    # save_cols_labels = []
    # for label in save_cols_less:
    #     save_cols_labels.append(''.join([name, '_', label]))

    airport_df = airport_df[save_cols_less]
    #airport_df.columns = save_cols_labels
    airport_df.columns = save_cols_less

    return airport_df
