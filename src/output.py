import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from plotting_scripts import feat_importance_plot, output_histograms

if __name__=='__main__':
    # load data
    (y_true, y_hat) = pickle.load( open( 'pkl/aspen_d2_slabwet_labeled_out.p', 'rb'))

    #output_histograms(y_true, y_hat)

    # fit distibution
    true_slab = y_true[0]
    pred_slab = y_hat[0]
    # define x, xx
    xt = true_slab
    xp = pred_slab

    xx = np.linspace(min(true_slab), max(true_slab), 300)

    ''' plot beta distributions and histograms '''
    a = [1.3, 1.6]
    b = [7, 8]
    dist_t = stats.beta(a[0], b[0], loc=0, scale=8)
    dist_p = stats.beta(a[1] ,b[1], loc=0, scale=7)

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].hist(xt, bins=20, normed=True)
    ax[0].plot(xx, dist_t.pdf(xx), '-k',
            label=r'beta($\alpha$ = {}, $\beta$ = {})'.format(a[0],b[0]))
    ax[0].set_title('true')
    ax[0].legend()

    ax[1].hist(xp, bins=20, normed=True)
    ax[1].plot(xx, dist_p.pdf(xx), '-k',
            label=r'beta($\alpha$ = {}, $\beta$ = {})'.format(a[1],b[1]))
    ax[1].set_title('predicted')
    ax[1].legend()

    plt.show()


''' dump '''
    # hh, edges = np.histogram(x, 30, density=True)
    #
    # xx = np.linspace(edges[0], edges[-1], 300)
    #
    # # normal
    # m,s = stats.norm.fit(hh)
    # pdf_n = stats.norm.pdf(xx, m, s)
    #
    # # gamma
    # a,b,g = stats.gamma.fit(hh)
    # pdf_g = stats.gamma.pdf(xx, a, b, g)
    #
    # # beta
    # aa, bb, loc, scale = stats.beta.fit(hh, loc=0, scale=edges[-1])
    # dist_b = stats.beta(aa, bb, loc=0, scale=edges[-1])
    # pdf_b = dist_b.pdf(xx)
    #
    # fig, ax = plt.subplots()
    # plt.hist(x, bins=30, normed=True)
    # plt.plot(edges[1:],hh,'ok')
    # plt.plot(xx, pdf_n, label='norm')
    # plt.plot(xx, pdf_g, label='gamma')
    # plt.plot(xx, pdf_b, label='beta')
    # plt.legend()
    # plt.show()
