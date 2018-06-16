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
                label='actual # of events'.format(case[0]))
    #h2 = ax1.fill(test_df.index, test_df.N_WET, c_t[1],
    #            label='actual {}'.format(case[1]))
    ax1.tick_params(axis='x', rotation='auto')
    ax1.set_ylim([0,15.1])
    ax1.set_yticks([0,5,10,15])

    ax2 = ax1.twinx()
    h3 = ax2.fill(test_df.index,slab_proba,c_p[0],
                alpha = 0.3,
                label='predicted p({})'.format(case[0]))
    h4 = ax2.fill(test_df.index,wet_proba,c_p[1],
                alpha = 0.3,
                label='predicted p({})'.format(case[1]))
    ax2.set_ylabel('probability')
    ax2.set_ylim([0,1.1])

    hh = h1 + h3 + h4
    h_labels = [h.get_label()  for h in hh]
    ax2.legend(hh, h_labels, loc=0)
    plt.title('Predictions: Aspen Zone')
    plt.tight_layout()
    plt.show()

def results_ts_plot_2(df, slab_proba, wet_proba):
    # get train, test avy data
    splitdate = pd.to_datetime('2016-06-01')
    train_df = df[df.index <= splitdate]
    test_df = df[df.index > splitdate]

    # binary for any avalanche
    test_df['AVY'] = test_df.SLAB + test_df.WET
    test_df.AVY.apply(lambda x: 1 if x == 2 else x)
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
    ax1.bar(test_df.index, test_df.AVY, color=c_t[0], alpha=0.2,
                label='avalanche occurence'.format(case[0]), width=1)
    #h2 = ax1.fill(test_df.index, test_df.N_WET, c_t[1],
    #            label='actual {}'.format(case[1]))
    #ax1.tick_params(axis='x', rotation='auto')
    #ax1.set_ylim([0,1])
    #ax1.set_yticks([0,5,10,15])

    #ax2 = ax1.twinx()
    ax1.fill(test_df.index,slab_proba,c_p[0],
                alpha = 0.3,
                label='predicted p({})'.format(case[0]))
    ax1.fill(test_df.index,wet_proba,c_p[1],
                alpha = 0.3,
                label='predicted p({})'.format(case[1]))
    ax1.set_ylabel('probability')
    ax1.set_ylim([0,1.1])

    ax1.legend()

    #hh = h3 + h4
    #h_labels = [h.get_label()  for h in hh]
    #ax2.legend(hh, h_labels, loc=0)
    #plt.legend('actual, ps, pw')
    plt.title('Predictions: Aspen Zone')
    plt.tight_layout()
    plt.show()

def results_ts_plot_3(df, slab_proba, wet_proba):
    # get train, test avy data
    splitdate = pd.to_datetime('2016-06-01')
    train_df = df[df.index <= splitdate]
    test_df = df[df.index > splitdate]

    # binary for any avalanche
    test_df['AVY'] = test_df.SLAB + test_df.WET
    test_df.AVY.apply(lambda x: 1 if x == 2 else x)
    # n_avy for test, train, divided into slab, wet
    n_avy_train = train_df.N_AVY
    n_avy_test = test_df.N_AVY
    test_df['N_SLAB'] = test_df.N_AVY * test_df.SLAB
    test_df['N_WET'] = test_df.N_AVY * test_df.WET

    case = ['SLAB', 'WET']
    c_t = ['r','orange']
    c_p = ['b','g']

    # plot
    fig, ax = plt.subplots(2,1)
    ax[0].set_ylabel('daily # of avalanches')
    # here (N_SLAB, N_WET) or N_AVY
    ax[0].bar(test_df.index, test_df.N_AVY, color=c_t[0], alpha=0.3,
                label='avalanche occurence'.format(case[0]), width=1)
    ax[0].legend()
    #h2 = ax1.fill(test_df.index, test_df.N_WET, c_t[1],
    #            label='actual {}'.format(case[1]))
    #ax1.tick_params(axis='x', rotation='auto')
    ax[0].set_ylim([0,12])
    ax[0].set_yticks([0,5,10,15])

    #ax2 = ax1.twinx()
    ax[1].bar(test_df.index,slab_proba, color=c_p[0],
                alpha = 0.3,
                label='predicted p({})'.format(case[0]))
    ax[1].bar(test_df.index,wet_proba, color=c_p[1],
                alpha = 0.3,
                label='predicted p({})'.format(case[1]))
    ax[1].set_ylabel('probability')
    ax[1].set_ylim([0,1.1])

    ax[1].legend()

    #hh = h3 + h4
    #h_labels = [h.get_label()  for h in hh]
    #ax2.legend(hh, h_labels, loc=0)
    #plt.legend('actual, ps, pw')
    plt.title('Predictions: Aspen Zone')
    plt.tight_layout()
    plt.show()

def results_ts_plot_4(df, slab_proba, wet_proba):
    # get train, test avy data
    splitdate = pd.to_datetime('2016-06-01')
    train_df = df[df.index <= splitdate]
    test_df = df[df.index > splitdate]

    # binary for any avalanche
    test_df['AVY'] = test_df.SLAB + test_df.WET
    test_df.AVY.apply(lambda x: 1 if x == 2 else x)
    # n_avy for test, train, divided into slab, wet
    n_avy_train = train_df.N_AVY
    n_avy_test = test_df.N_AVY
    test_df['N_SLAB'] = test_df.N_AVY * test_df.SLAB
    test_df['N_WET'] = test_df.N_AVY * test_df.WET

    case = ['SLAB', 'WET']
    c_t = ['#ffb366','orange']
    c_p = ['#7070db','#2eb8b8']

    months = ['oct','nov','dec','jan','feb','mar','apr','may','jun','jul','aug','sep']

    # plot
    fig, ax = plt.subplots()
    #ax.set_ylabel('daily # of avalanches')
    # here (N_SLAB, N_WET) or N_AVY
    ax.bar(test_df.index, test_df.AVY*2, color=c_t[0], alpha=0.6,
                width=1,
                label='avalanche'.format(case[0]))
    ax.legend()
    #h2 = ax1.fill(test_df.index, test_df.N_WET, c_t[1],
    #            label='actual {}'.format(case[1]))
    #ax1.tick_params(axis='x', rotation='auto')
    #ax[0].set_ylim([0,12])
    #ax[0].set_yticks([0,5,10,15])

    #ax2 = ax1.twinx()
    ax.fill(test_df.index,slab_proba, color=c_p[0],
                alpha = 0.8,
                label='p({})'.format(case[0]))
    ax.fill(test_df.index,wet_proba, color=c_p[1],
                alpha = 0.8,
                label='p({})'.format(case[1]))
    ax.set_ylabel('probability')
    ax.set_ylim([0,1.2])
    ax.set_yticks(np.linspace(0,1,6))
    #ax.set_xticks(np.linspace(1,300,12),months)

    ax.legend(loc=0)

    ax =plt.gca()
    #ax.set_facecolor('k')

    #hh = h3 + h4
    #h_labels = [h.get_label()  for h in hh]
    #ax2.legend(hh, h_labels, loc=0)
    #plt.legend('actual, ps, pw')
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
    results_ts_plot_4(df, slab_proba, wet_proba)
