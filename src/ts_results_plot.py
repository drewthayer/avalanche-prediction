import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

def results_ts_plot(df, slab_proba, wet_proba):
    # get train, test avy data
    splitdate = pd.to_datetime('2016-06-01')
    train_df = df[df.index <= splitdate]
    test_df = df[df.index > splitdate]

    # n_avy for test, train, divided into slab, wet
    n_avy_train = train_df.N_AVY
    n_avy_test = test_df.N_AVY
    test_df['N_SLAB'] = test_df.N_AVY * test_df.SLAB
    test_df['N_WET'] = test_df.N_AVY * test_df.WET

    case = ['SLAB', 'WET']
    c_t = ['r','orange']
    c_p = ['b','g']

    # plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('date')
    ax1.set_ylabel('daily # of avalanches')
    # here (N_SLAB, N_WET) or N_AVY
    h1 = ax1.fill(test_df.index, test_df.N_AVY, c_t[0],
                label='actual'.format(case[0]))
    #h2 = ax1.fill(test_df.index, test_df.N_WET, c_t[1],
    #            label='actual {}'.format(case[1]))
    ax1.tick_params(axis='x', rotation='auto')
    ax1.set_ylim([0,15.1])
    ax1.set_yticks([0,5,10,15])

    ax2 = ax1.twinx()
    h3 = ax2.fill(test_df.index,slab_proba,c_p[0],
                alpha = 0.3,
                label='predicted {}'.format(case[0]))
    h4 = ax2.fill(test_df.index,wet_proba,c_p[1],
                alpha = 0.3,
                label='predicted {}'.format(case[1]))
    ax2.set_ylabel('probability')
    ax2.set_ylim([0,1.1])

    hh = h1 + h3 + h4
    h_labels = [h.get_label()  for h in hh]
    ax2.legend(hh, h_labels, loc=0)
    plt.title('Predictions: Aspen Zone')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    # load data
    df = pickle.load( open( 'pkl/aspen_d2_imputemean_alldays.p', 'rb'))
    # load model results
    y_true_l, y_hat_l, y_proba_l, feat_list, test_timestamps = pickle.load(
            open( 'pkl/aspen_gbc_smoted_scaled_output.p', 'rb'))

    # probabilities
    slab_proba = y_proba_l[0][:,1]
    wet_proba = y_proba_l[1][:,1]

    # ts plot
    results_ts_plot(df, slab_proba, wet_proba)
