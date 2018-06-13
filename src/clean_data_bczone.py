import os
import pandas as pd
# my scripts
from cleaning_scripts import clean_snow_data, remove_airtemp_outliers, clean_airport_data
from sqlite3_scripts import connect_to_sql, create_table_sql, write_pandas_to_sql

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
    snotel_df.drop('dt', axis=1, inplace=True)

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

    return airport_df

def read_zone_data_to_sql(zonename, zonedata, snotel_dir, lcd_dir):
    # read airport wind speed data to df
    airport_df = read_and_concat_airport(lcd_dir, zonedata[zonename][1])

    # read snotel data to df
    snotel_df = read_and_concat_snotel(snotel_dir, zonedata[zonename][0])

    #write to sql db
    named_dfs = {'airport':airport_df, 'snotel':snotel_df}
    db = current + '/../data/data-{}.db'.format(zonename)
    conn = connect_to_sql(db)
    for tablename in named_dfs:
        write_pandas_to_sql(conn, tablename, named_dfs[tablename])
    conn.close()

if __name__=='__main__':
    # define paths
    current = os.getcwd()
    snotel_dir = ''.join([current,'/../data/data-snotel/'])
    lcd_dir = ''.join([current,'/../data/data-LCD/'])

    # stations and airport files for different zones
    stations_aspen = ['618_mcclure_pass',
                        '669_north_lost_trail',
                        '737_schofield_pass',
                        '542_independence_pass',
                        '369_brumley',
                        '547_ivanhoe']
    airportfiles_aspen = ['aspen_pitkin_airport_20060101_current.csv',
                   'leadville_lake_airport_20090101.csv',
                   'telluride_airport_20090101_20180417.csv']

    stations_nsj = ['713_red_mtn_pass',
                        '538_idarado']
    airportfiles_nsj = ['montrose_airport_20060101_20081231.csv',
                        'montrose_airport_20090101_20180405.csv']
    # zone data dictionary:
    zonedata = {'aspen': [stations_aspen, airportfiles_aspen],
                'nsj': [stations_nsj, airportfiles_nsj]}

    # read data, write to sql, input from command line 
    zone = sys.argv[1]
    read_zone_data_to_sql('nsj', zonedata[zone], snotel_dir, lcd_dir)
