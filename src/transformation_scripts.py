import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def water_year_month(month):
    if month >= 10:
        month2 = month - 9
    else:
        month2 = month + 3
    return month2

def water_year_day(day):
    if day >= 273:
        day2 = day - 273
    else:
        day2 = day + 92
    return day2

def oversample(data_df, colname, n=4):

    ''' oversample data based on column name
    input: pandas dataframe
    output: pandas dataframe, shuffled
    '''

    mini_dfs = []
    for i in range(n+1):
        mini_dfs.append(data_df[data_df[colname] == i])

    counts = {}
    for x in range(n+1):
        counts[x] = data_df[data_df[colname] == x].count().max()

    factors = {}
    for x in range(x+1): # no 7 b/c 0
        factors[x] = counts[0]//counts[x]

    # concatenate to dataframe
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

def div_count_pos_neg(X, y):
    """Helper function to divide X & y into positive and negative classes
    and counts up the number in each.
    Parameters
    ----------
    X : ndarray - 2D
    y : ndarray - 1D
    Returns
    -------
    negative_count : Int
    positive_count : Int
    X_positives    : ndarray - 2D
    X_negatives    : ndarray - 2D
    y_positives    : ndarray - 1D
    y_negatives    : ndarray - 1D
    """
    negatives, positives = y == 0, y == 1
    negative_count, positive_count = np.sum(negatives), np.sum(positives)
    X_positives, y_positives = X[positives], y[positives]
    X_negatives, y_negatives = X[negatives], y[negatives]
    return negative_count, positive_count, X_positives, \
           X_negatives, y_positives, y_negatives

def smote(X, y, tp, k=None):
    """Generates new observations from the positive (minority) class.
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf
    Notes: Currently the KNeighborsClassifier will throw a warning when calling
           to the kneighbors method. Appears to be happening in sklearn not
           this usage of it.
    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - [0, 1], target proportion of positive class observations
    Returns
    -------
    X_smoted : ndarray - 2D
    y_smoted : ndarray - 1D
    """
    if tp < np.mean(y):
        return X, y
    if k is None:
        k = int(len(X) ** 0.5)

    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_pos, y_pos)
    neighbors = knn.kneighbors(return_distance=False)

    positive_size = (tp * neg_count) / (1 - tp)
    smote_num = int(positive_size - pos_count)

    rand_idxs = np.random.randint(0, pos_count, size=smote_num)
    rand_nghb_idxs = np.random.randint(0, k, size=smote_num)
    rand_pcts = np.random.random((smote_num, X.shape[1]))
    smotes = []
    for r_idx, r_nghb_idx, r_pct in zip(rand_idxs, rand_nghb_idxs, rand_pcts):
        rand_pos, rand_pos_neighbors = X_pos[r_idx], neighbors[r_idx]
        rand_pos_neighbor = X_pos[rand_pos_neighbors[r_nghb_idx]]
        rand_dir = rand_pos_neighbor - rand_pos
        rand_change = rand_dir * r_pct
        smoted_point = rand_pos + rand_change
        smotes.append(smoted_point)

    X_smoted = np.vstack((X, np.array(smotes)))
    y_smoted = np.concatenate((y, np.ones((smote_num,))))
    return X_smoted, y_smoted
