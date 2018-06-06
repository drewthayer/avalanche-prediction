import os
import pandas as pd
import sqlite3

import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        conn.close()

def connect_to_db(db_file):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
        return conn
    except Error as e:
        print(e)
    #finally:
    #    conn.close()

def create_table(conn, tablename):
    cursor = conn.cursor()

    sql_command = """
    CREATE TABLE IF NOT EXISTS tablename (
    date DATE,
    bc_zone VARCHAR(20),
    type VARCHAR(3),
    dsize FLOAT,
    n_avy INTEGER);"""

    cursor.execute(sql_command)
    conn.close()

def write_pandas_to_sql(conn, tablename, df):
    df.to_sql(tablename, conn, if_exists='replace')
    conn.close()


def read_from_db(conn, query):
    cursor = conn.cursor()

    cursor.execute(query)
    result = cursor.fetchall()
    for r in result:
        print(r)

    conn.close()

# if __name__ == '__main__':
#     create_connection("C:\\sqlite\db\pythonsqlite.db")

if __name__=='__main__':
    # paths
    current = os.getcwd()
    clean_dir = ''.join([current,'/../data/data-clean/'])
    # load data
    avy_df = pd.read_csv(clean_dir + 'avy_data.csv')
    airport_df = pd.read_csv(clean_dir + 'airport_data_NSanJuan.csv')
    snotel_df = pd.read_csv(clean_dir + 'snotel_data_NSanJuan.csv')

    '''write to sql db'''
    db = current + '/../data/data-clean-db/avalanche.db'
    tablename = 'avalanche'
    conn = connect_to_db(db)
    create_table(conn, tablename)

    # write df to sql
    conn = connect_to_db(db)
    write_pandas_to_sql(conn, tablename, avy_df)

    ''' read from sql db '''
    query = """SELECT datetime, Dsize from avalanche LIMIT 5"""
    conn = connect_to_db(db)
    read_from_db(conn, query)
