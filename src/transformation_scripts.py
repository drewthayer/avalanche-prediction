import pandas as pd

def oversample(data_df):

    ''' oversample days with many avalanches '''

    # frequency of avys/day:
    # n = 2004
    #  {0: 1533, 1: 381, 2: 42, 3: 20, 4: 7, 5: 12, 6: 8, 7: 0}

    avy0 = data_df[data_df.D2_up == 0]
    avy1 = data_df[data_df.D2_up == 1]
    avy2 = data_df[data_df.D2_up == 2]
    avy3 = data_df[data_df.D2_up == 3]
    avy4 = data_df[data_df.D2_up == 4]
    avy5 = data_df[data_df.D2_up == 5]
    avy6 = data_df[data_df.D2_up == 6]

    n_avy = [0,1,2,3,4,5,6]
    counts = {}
    for n in n_avy:
        counts[n] = data_df[data_df.D2_up == n].count().max()

    # duplication factors
    # {2: 9, 3: 19, 4: 54, 5: 31, 6: 47}
    factors = {}
    for n in n_avy: # no 7 b/c 0
        factors[n] = counts[0]//counts[n]

    # concatenate to dataframe
    mini_dfs = [avy0, avy1, avy2, avy3, avy4, avy5, avy6]
    frames = []
    for n in factors.keys():
        i = 0
        while i <= factors[n]:
            frames.append(mini_dfs[n])
            i += 1

    df = pd.concat(frames, axis=0)

    # random shuffle
    df_shuffle = df.copy()
    df_shuffle.set_index(np.random.permutation(df_shuffle.index), inplace=True)

    return df_shuffle
