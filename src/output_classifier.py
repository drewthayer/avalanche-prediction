import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from ROC import ROC

from plotting_scripts import feat_importance_plot, output_histograms_classification

if __name__=='__main__':
    # load data
    y_true_l, y_hat_l, y_proba_l, feat_list, test_timestamps = pickle.load(
            open( 'pkl/aspen_d2_rfc_best_output.p', 'rb'))

    # probabilities
    slab_proba = y_proba_l[0][:,1]
    wet_proba = y_proba_l[1][:,1]
    sum_proba = sum([x[:,1] for x in y_proba_l])
    sum_proba_n = sum_proba/max(sum_proba)
    h_labels = ['p(slab)','p(wet)','$\Sigma$ slab, wet']

    # output histograms
    output_histograms_classification(slab_proba, wet_proba, sum_proba_n, h_labels)

    # output histograms: zero removed
    output_histograms_classification(
        slab_proba[slab_proba > 0.01],
        wet_proba[wet_proba > 0.01],
        sum_proba_n[sum_proba_n > 0.01],
        h_labels)

    # sum true, predicted, probs
    sum_true = sum(x for x in y_true_l)
    sum_true.apply(lambda x: 1 if x == 2 else x) # turn shared days (2) to 1
    sum_hat = sum(y_hat_l)
    y_true_l.append(sum_true)
    y_hat_l.append(sum_hat)
    proba_l = [slab_proba, wet_proba]
    proba_l.append(sum_proba_n)

    # roc, metrics:
    labels = ['slab', 'wet', 'sum']
    for y_true, y_hat, y_proba, name in zip(y_true_l, y_hat_l, proba_l, labels):
        print(name)
        roc = ROC()
        roc.fit(y_true, y_hat)
        print(f'acc: {roc.accuracy}')
        print(f'prec: {roc.precision}')
        print(f'rec: {roc.recall}')

        # roc plot
        roc.roc_plot(y_true, y_proba,'b','{}'.format(name),'{}_roc.png'.format(name))

        # apr plot
        roc.a_r_p_plot(y_true, y_proba)
