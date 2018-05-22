import pandas as pd
import numpy as np
import pickle
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

from transformation_scripts import water_year_day

def kde_probabilities(dataframe, feature_vals, feature_col, target_col, xx, smooth):
    probs_list = []
    for vals in feature_vals:
        X = dataframe[np.in1d(avy_df[feature_col], vals)][target_col]
        density = gaussian_kde(X)
        density.covariance_factor = lambda : smooth
        density._compute_covariance()
        probs = density(xx)
        probs_list.append(probs)
    return probs_list

if __name__=='__main__':
    avy_df = pd.read_csv('../data/data-clean/avy_data.csv')

    # add DOY water year column
    dt = pd.to_datetime(avy_df.Date)
    doy = dt.apply(lambda x: x.timetuple().tm_yday)
    avy_df['DOY'] = doy.apply(lambda x: water_year_day(x))

    # kde probabilities
    feature_vals = [['SS','HS','L'], ['WL']]
    feature_col = 'Type'
    target_col = 'DOY'
    days = np.linspace(1,365,365)
    smooth = 0.35
    probs_list = kde_probabilities(avy_df, feature_vals, feature_col, target_col, days, smooth)

    # plot
    fig, ax = plt.subplots()
    plt.fill(probs_list[0], 'b', label='slab', alpha=0.5)
    plt.fill(probs_list[1], 'g', label='wet', alpha=0.5)
    plt.legend()
    plt.xlabel('day of water year')
    plt.ylabel('probability')
    plt.show()

    # save dictionary of day, probability
    p_slab = dict(zip(days,probs_list[0]))
    p_wet = dict(zip(days,probs_list[1]))

    # write dictionaries (as tuple) to pickle
    pickle.dump((p_slab, p_wet), open('pkl/kde_probs.p', 'wb'))
