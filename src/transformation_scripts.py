import pandas as pd
import numpy as np

def oversample(data_df, colname, n=4):

    ''' oversample data based on column name
    input: pandas dataframe
    output: pandas dataframe, shuffled
    '''

    mini_dfs = []
    for i in range(n+1):
        mini_dfs.append(data_df[data_df[colname] == i])


    # avy0 = data_df[data_df[colname] == 0]
    # avy1 = data_df[data_df[colname] == 1]
    # avy2 = data_df[data_df[colname] == 2]
    # avy3 = data_df[data_df[colname] == 3]
    # avy4 = data_df[data_df[colname] == 4]
    # avy5 = data_df[data_df[colname] == 5]
    # avy6 = data_df[data_df[colname] == 6]

    #n_avy = [0,1,2,3,4,5,6]
    counts = {}
    for x in range(n+1):
        counts[x] = data_df[data_df[colname] == x].count().max()

    # duplication factors
    # {2: 9, 3: 19, 4: 54, 5: 31, 6: 47}
    factors = {}
    for x in range(x+1): # no 7 b/c 0
        factors[x] = counts[0]//counts[x]

    # concatenate to dataframe
    #mini_dfs = [avy0, avy1, avy2, avy3, avy4, avy5, avy6]
    frames = [data_df]
    for x in factors.keys():
        i = 0
        while i <= factors[x]:
            frames.append(mini_dfs[x])
            i += 1

    df = pd.concat(frames, axis=0)

    # random shuffle
    df_shuffle = df.copy()
    df_shuffle.set_index(np.random.permutation(df_shuffle.index), inplace=True)

    return df_shuffle, counts, factors
