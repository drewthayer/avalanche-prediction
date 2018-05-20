import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def binary_prob_plots(df):
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].plot(df.P_SLAB, df.SLAB, 'ok')
    ax[0].set_title('slab avalanche')
    ax[0].set_xlabel('p(slab)')
    ax[0].set_ylabel('events')
    ax[1].plot(df.P_WET, df.WET, 'ok')
    ax[1].set_title('wet avalanche')
    ax[1].set_xlabel('p(wet)')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    # load data
    df = pickle.load( open( "pkl/aspen_d2_imputemean.p", "rb"))

    # test for nans
    na_per_col = np.sum(df.isna())

    # probability plots
    binary_prob_plots(df)

    # types of features
    cols = list(df.columns)
    binary = ['SLAB', 'WET']
    continuous = [x for x in cols if x not in binary]
    # feature histograms: binary
    for col in binary:
        data = df[col].dropna()
        plt.hist(data)
        plt.title(col)
        plt.savefig('feature_hists/{}.png'.format(col),dpi=250)
        plt.close()

    # feature histograms: continuous,data w/o nan
    for col in continuous:
        data = df[col].dropna()
        plt.hist(data, bins=20)
        plt.title(col)
        plt.savefig('feature_hists/{}.png'.format(col),dpi=250)
        plt.close()
