import os
import pandas as pd
# my scripts
from scripts.sqlite3_scripts import connect_to_sql, write_pandas_to_sql

def read_caic_data(directory, filename):
    ''' reads caic data from .csv, adds datetime as index
    input: .csv file
    output: pandas df
    '''
    fname = directory + filename
    avy_df = pd.read_csv(fname)
    avy_df['datetime'] = pd.to_datetime(avy_df.Date)
    avy_df.set_index(avy_df.datetime, inplace=True)

    return avy_df

if __name__=='__main__':
    # define paths
    current = os.getcwd()
    caic_dir = ''.join([current,'/../data/data-caic/'])

    # read avalanche data to pandas df
    caic_file = 'CAIC_avalanches_2010-05-07_2018-04-10.csv'
    avy_df = read_caic_data(caic_dir, caic_file)
    avy_df.drop(['Date', 'datetime'], axis=1, inplace=True)

    # write to sql db
    db = current + '/../data/data-caic.db'
    conn = connect_to_sql(db)
    write_pandas_to_sql(conn, 'avalanche', avy_df)
    conn.close()
