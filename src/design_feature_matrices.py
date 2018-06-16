import pandas as pd
import pickle
import numpy as np
import os
import sys

from scripts.sqlite3_scripts import connect_to_sql
from scripts.transformation_scripts import smote, set_df_index_from_datetime
from scripts.modeling_scripts import train_test_split_time

if __name__=='__main__':
    zonename = sys.argv[1] # 'aspen' or 'nsj'
    case_select = sys.argv[2] # 'slab' or 'wet'
    # load data from sql as df, drop n_avy column
    current = os.getcwd()
    conn = connect_to_sql(current + '/../data/data-engineered.db')
    df = pd.read_sql("select * from {}".format(zonename), conn)

    # drop n_avy column and set index to datetime
    df.drop('N_AVY', axis=1, inplace=True)
    df = set_df_index_from_datetime(df, 'datetime')

    # fill na with zero in case any not imputed
    df.fillna(0, inplace=True)

    ''' case: list, length = # of cases, items = list of [a,b] and [b,a]
        case_params, list, same length as case, items = dicts of params
        '''
    cases = [['SLAB','WET'], ['WET','SLAB']]

    if case_select == 'slab':
        case = cases[0]
    elif case_select == 'wet':
        case = cases[1]

    data_df = df.copy() # copy to read all columns after dropping

    # drop other binary and probability column
    c_drop = [c for c in list(df.columns) if case[1] in c]
    data_df.drop(c_drop, axis=1, inplace=True)

    # train test split in time
    X_train, y_train, X_test, y_test = train_test_split_time(data_df,
        '2016-06-01', case[0])

    # oversample train data with SMOTE
    X_smoted, y_smoted = smote(X_train.values, y_train.values, 0.60, k=None)

    # write to pkl
    pickle.dump((X_smoted, y_smoted, X_test, y_test),
    open('feature-matrices/{}_{}_matrices.pkl'.format(zonename, case_select),
    'wb'))
