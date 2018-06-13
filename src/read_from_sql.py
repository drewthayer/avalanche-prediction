# select from sql
import os
import sqlite3
from sqlite3_scripts import connect_to_sql, read_from_sql, read_print_from_sql

if __name__=='__main__':
    current = os.getcwd()
    conn = connect_to_sql(current + '/../data/data-aspen.db')
    sql = '''
        SELECT DISTINCT airport
        from airport
    '''
    read_print_from_sql(conn, sql)

    out = read_from_sql(conn, sql)

    conn.close()

    # get table names
    con = sqlite3.connect(current + '/../data/data-aspen.db')
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cursor.fetchall())
