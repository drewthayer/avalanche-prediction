''' sqlite3 helper scripts '''
import sqlite3
from sqlite3 import Error

def connect_to_sql(db_file):
    """ create a database connection to a SQLite database
    input: path to database file
    output: sql connection
    """
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
        return conn
    except Error as e:
        print(e)

def create_table_sql(conn, tablename, sql):
    cursor = conn.cursor()
    cursor.execute(sql)

def write_pandas_to_sql(conn, tablename, df):
    df.to_sql(tablename, conn, if_exists='replace')


def read_from_sql(conn, query):
    cursor = conn.cursor()

    cursor.execute(query)
    result = cursor.fetchall()
    for r in result:
        print(r)

    conn.close()
