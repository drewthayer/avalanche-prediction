import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from ROC import ROC
from sklearn.metrics import precision_recall_curve

from plotting_scripts import feat_importance_plot, output_histograms_classification

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
    h1 = ax1.fill(test_df.index, test_df.N_SLAB, c_t[0],
                label='actual {}'.format(case[0]))
    h2 = ax1.fill(test_df.index, test_df.N_WET, c_t[1],
                label='actual {}'.format(case[1]))
    ax1.tick_params(axis='x', rotation='auto')
    ax1.set_ylim([0,15])

    ax2 = ax1.twinx()
    h3 = ax2.fill(test_df.index,slab_proba,c_p[0],
                alpha = 0.3,
                label='predicted {}'.format(case[0]))
    h4 = ax2.fill(test_df.index,wet_proba,c_p[1],
                alpha = 0.3,
                label='predicted {}'.format(case[1]))
    ax2.set_ylabel('probability')
    ax2.set_ylim([0,1.01])

    hh = h1 + h2 + h3 + h4
    h_labels = [h.get_label()  for h in hh]
    ax2.legend(hh, h_labels, loc=0)
    plt.title('Predictions: Aspen Zone')
    plt.tight_layout()
    plt.show()

def feature_importances(feat_list, label_list, color_list):
    feat_sort_l = []
    for item, label, color in zip(feat_list, label_list, color_list):
        names = list(item[0])
        importances = item[1]
        # make ordered list
        feat_sort_l.append(sorted(zip(names, importances),
                key=lambda x:abs(x[1]), reverse=True))
        # plot
        feat_importance_plot(names, importances, label, color, figsize=(6,6))
    return feat_sort_l

def idx_prediction(slab_proba, wet_proba, sum_proba, test_timestamps, idx):
    print('p(slab) = {:0.3f}'.format(slab_proba[idx]))
    print('p(wet) = {:0.3f}'.format(wet_proba[idx]))
    print('p(avalanche) = {:0.3f}'.format(sum_proba[idx]))
    actual = df[df.index == test_timestamps[idx]]['N_AVY']
    print('actual # of avalanches = {}'.format(actual[0]))

if __name__=='__main__':
    # load data
    df = pickle.load( open( 'pkl/aspen_d2_imputemean_alldays.p', 'rb'))
    # load model results
    y_true_l, y_hat_l, y_proba_l, feat_list, test_timestamps = pickle.load(
            open( 'pkl/aspen_gbc_smoted_scaled_output.p', 'rb'))

    # probabilities
    slab_proba = y_proba_l[0][:,1]
    wet_proba = y_proba_l[1][:,1]
    sum_proba = sum([x[:,1] for x in y_proba_l])
    sum_proba_n = sum_proba/max(sum_proba)
    h_labels = ['p(slab)','p(wet)','$\Sigma$ slab, wet']

    # prediction, actual for a particular day
    idx = 105 # good examples: 13, 14, 105
    idx_prediction(slab_proba, wet_proba, sum_proba, test_timestamps, idx)

    # output histograms
    output_histograms_classification(slab_proba, wet_proba, sum_proba_n, h_labels)

    # output histograms: zero removed
    output_histograms_classification(
        slab_proba[slab_proba > 0.01],
        wet_proba[wet_proba > 0.01],
        sum_proba_n[sum_proba_n > 0.01],
        h_labels)

    # sum true, predicted, probs
    sum_true = y_true_l[0] + y_true_l[1]
    sum_true.apply(lambda x: 1 if x == 2 else x) # turn shared days (2) to 1
    sum_hat = sum(y_hat_l)
    sum_hat[sum_hat > 1] = 1 # turn shared days to 1
    # append to lists
    y_true_l.append(sum_true)
    y_hat_l.append(sum_hat)
    proba_l = [slab_proba, wet_proba]
    proba_l.append(sum_proba)

    # ts plot
    results_ts_plot(df, slab_proba, wet_proba)

    # feature importances
    label_list = ['gbc: slab', 'gbc: wet']
    color_list = ['b','g']
    feat_sort_l = feature_importances(feat_list, label_list, color_list)

    # accuracy, precision, recall metrics:
    labels = ['slab', 'wet', 'combined']
    colors = ['b', 'g', 'purple']
    fpr_tpr_l = []
    for y_true, y_hat, y_proba, name in zip(y_true_l, y_hat_l, proba_l, labels):
        print(name)
        roc = ROC()
        roc.fit(y_true, y_hat)
        print(f'acc: {roc.accuracy}')
        print(f'prec: {roc.precision}')
        print(f'rec: {roc.recall}')

        # roc TPR, FPR
        fpr, tpr = roc.calc_roc(y_true, y_proba)
        fpr_tpr_l.append((fpr, tpr))

        # apr plot
        precision, recall, thresholds = precision_recall_curve(y_true.values,
            y_proba, pos_label=1)
        thresholds = np.append(thresholds, 1)
        plt.plot(thresholds, recall, 'b--', label='recall')
        plt.plot(thresholds, precision, 'g--', label='precision')
        plt.legend()
        plt.show()

        roc.a_r_p_plot(y_true, y_proba)


    # roc plot
    fig, ax = plt.subplots()
    for item, color, label in zip(fpr_tpr_l, colors, labels):
        plt.plot(item[0], item[1], color=color, label=label)
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.show()

    # is model adding value beyond p_slab, p_wet
    n = len(slab_proba)
    pps = df.P_SLAB[-n:]
    ppw = df.P_WET[-n:]
    plt.plot(pps, slab_proba, 'ob', label='slab')
    plt.plot(ppw, wet_proba, 'og', label='wet')
    plt.xlabel('p() as fn of d-o-y')
    plt.ylabel('predicted p()')
    plt.legend()
    plt.show()

    # n_avy histogram
    plt.hist(df.N_AVY[df.N_AVY > 0], 20)
    plt.yscale('log', nonposy='clip')
    plt.xlabel('# of avalanches')
    plt.ylabel('count')
    plt.show()
