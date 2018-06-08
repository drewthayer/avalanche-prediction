# select from sql
import os
from sqlite3_scripts import connect_to_sql, create_table_sql, read_from_sql

if __name__=='__main__':
    current = os.getcwd()
    conn = connect_to_sql(current + '/../data/data-clean.db')
    sql = '''
        SELECT DISTINCT airport 
        from airport
    '''
    read_from_sql(conn, sql)
