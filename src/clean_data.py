import os
import pandas as pd
import sqlite3
from sqlite3 import Error
from cleaning_scripts import clean_snow_data, remove_airtemp_outliers, clean_airport_data

def read_and_concat_snotel(directory, station_list):
    cols_raw = ['dt_string', 'swe_start_in', 'precip_start_in', 'airtemp_max_F', 'airtemp_min_F',
                    'airtemp_mean_F', 'precip_incr_in']
    data_list = []
    for station in station_list:
        snow_df = pd.read_csv(directory + '/' + 'snotel_{}.csv'.format(station),header=58)
        snow_df.columns = cols_raw
        snow_df = clean_snow_data(snow_df)
        #snow_df = remove_airtemp_outliers(snow_df)
        # add station name column
        snow_df['station'] = station
        # put station column first
        colnames = list(snow_df.columns)
        snow_df = snow_df[[colnames[-1]] + colnames[:-1]]
        data_list.append(snow_df)

    snotel_df = pd.concat(data_list, axis=0)

    return snotel_df

def read_and_concat_airport(directory, airportfiles):
    airport_list = []
    for filename in airportfiles:
        file = directory + filename
        name = filename.split('_')[0]
        df = pd.read_csv(file)
        df = clean_airport_data(df, name)
        df['airport'] = name
        airport_list.append(df)

    airport_df = pd.concat(airport_list, axis=0)

    return airport_df, name

def read_caic_data(directory, filename):
    fname = directory + filename
    avy_df = pd.read_csv(fname)
    avy_df['datetime'] = pd.to_datetime(avy_df.Date)
    avy_df.set_index(avy_df.datetime, inplace=True)

    return avy_df


if __name__=='__main__':
    # paths
    current = os.getcwd()
    snotel_dir = ''.join([current,'/../data/data-snotel/'])
    lcd_dir = ''.join([current,'/../data/data-LCD/'])
    caic_dir = ''.join([current,'/../data/data-caic/'])
    clean_dir = ''.join([current,'/../data/data-clean/'])

    # read snotel data
    stationnames_aspen = ['618_mcclure_pass',
                        '669_north_lost_trail',
                        '737_schofield_pass',
                        '542_independence_pass',
                        '369_brumley',
                        '547_ivanhoe']

    stationnames_nsj = ['713_red_mtn_pass',
                        '538_idarado',]

    snotel_df = read_and_concat_snotel(snotel_dir, stationnames_nsj)

    # read airport wind speed data
    #airportfiles_aspen = ['aspen_pitkin_airport_20060101_current.csv',
#                    'leadville_lake_airport_20090101.csv',
#                    'telluride_airport_20090101_20180417.csv']
    airportfiles_nsj = ['montrose_airport_20060101_20081231.csv',
                        'montrose_airport_20090101_20180405.csv']
    airport_df, name = read_and_concat_airport(lcd_dir, airportfiles_nsj)

    # avalanche data
    caic_file = 'CAIC_avalanches_2010-05-07_2018-04-10.csv'
    avy_df = read_caic_data(caic_dir, caic_file)

    ''' write data to file '''
    # write clean data to new csv
    # snotel_df.to_csv(clean_dir + 'snotel_data_NSanJuan.csv')
    # airport_df.to_csv(clean_dir + 'airport_data_NSanJuan.csv')
    #avy_df.to_csv(clean_dir + 'avy_data.csv')

    # to append:
    #with open(clean_dir + 'airport_data.csv', 'a') as f:
    #    airport_df.to_csv(f, header=False)
