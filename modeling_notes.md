### modeling notes

5/14/18 ran with first iteration of features from LCC paper
 - major increase in performance!
 - figs/prelim_imbalanced
 - DSUM is the most important feature by a LOT
    - without DSUM:
      - rfr out-of-bag train score = 0.058
      - rfr test rmse = 3.644
    - with DSUM:
      - rfr out-of-bag train score = 0.677
      - rfr test rmse = 2.030

 - ok, so that DSUM feature = the target and should have been removed !
   - results now good, but not ridiculous
   - figs/prelim_imbalanced_2
   - rfr out-of-bag train score = 0.053
   - rfr test rmse = 3.707

  - oversample for days 0-6:
   - rfr out-of-bag train score = 0.790
   - rfr test rmse = 3.767

oversamp to 10:
rfr out-of-bag train score = 0.935
rfr test rmse = 3.605

In [120]: counts
Out[120]: {0: 154, 1: 119, 2: 34, 3: 23, 4: 8, 5: 6, 6: 9, 7: 6, 8: 2, 9: 0, 10: 1}

In [121]: factors
Out[121]: {0: 1, 1: 1, 2: 4, 3: 6, 4: 19, 5: 25, 6: 17, 7: 25, 8: 77, 9: 0, 10: 154}

oversamp to 20:
fr out-of-bag train score = 0.997
rfr test rmse = 4.160

In [125]: data_df.shape
Out[125]: (510, 20)

In [126]: train_shuffle.shape
Out[126]: (3276, 20)


In [124]: factors
Out[124]:
{0: 1,
 1: 1,
 2: 4,
 3: 6,
 4: 19,
 5: 25,
 6: 17,
 7: 25,
 8: 77,
 9: 0,
 10: 154,
 11: 154,
 12: 0,
 13: 154,
 14: 154,
 15: 154,
 16: 0,
 17: 154,
 18: 154,
 19: 154,
 20: 0}

#### predict d3
rfr out-of-bag train score = 0.997
rfr test rmse = 0.811

rmse not too bad, but lots of predictions when none exist
factors: In [4]: factors
Out[4]:
{0: 1,
 1: 11,
 2: 54,
 3: 81,
 4: 325,
 5: 325,
 6: 325,
 7: 0,
 8: 162}

### add p(slab), p(wet), kde from whole dataset
rfr out-of-bag train score = 0.983
rfr test rmse = 3.633

### am i including enough days?
In [126]: merge_all.DOY.unique().shape
Out[126]: (181,)


### slab/wet predictions (no class balancing)
In [3]: run run_model_slab_wet.py
rfr out-of-bag train score = 0.118
rfr test rmse = 3.626

In [4]: run run_model_slab_wet.py
rfr out-of-bag train score = 0.209
rfr test rmse = 2.190

### slab/wet predictions (class balancing )
__wet__:
oversamp to 20:
rfr out-of-bag train score = 1.000
rfr test rmse = 2.385

oversamp to 15:
rfr out-of-bag train score = 0.997
rfr test rmse = 2.012

oversamp to 10:
rfr out-of-bag train score = 0.965
rfr test rmse = 1.897

oversamp to 6:
rfr out-of-bag train score = 0.965
rfr test rmse = 1.893

no oversampling:
rfr out-of-bag train score = 0.204
rfr test rmse = 2.194

__slab__:
no oversampling:
rfr out-of-bag train score = 0.133
rfr test rmse = 3.620 (similar to just adding probabilities)

oversample to n = 6
rfr out-of-bag train score = 0.866
rfr test rmse = 3.585

oversample to n = 10
rfr out-of-bag train score = 0.964
rfr test rmse = 3.572

oversample to n = 15
rfr out-of-bag train score = 0.989
rfr test rmse = 3.529

oversample to n = 20
rfr out-of-bag train score = 0.999
rfr test rmse = 3.996 (BUT gets high numbers)

oversample to n = 25
rfr out-of-bag train score = 0.999
rfr test rmse = 3.993

#### two models: one for slab, one for wet
case: SLAB
oversample to n = 15
out-of-bag train score = 0.989
test rmse = 3.580 (compare to 3.63 for just adding ps, one model )
case: WET
oversample to n = 6
out-of-bag train score = 0.958
test rmse = 2.300

#### two models with imputed mean: worse than imputed zero
case: SLAB
oversample to n = 15
out-of-bag train score = 0.989
test rmse = 3.616
case: WET
oversample to n = 6
out-of-bag train score = 0.958
test rmse = 2.304

#### classifier
class balance: In [50]: 212/510
Out[50]: 0.41568627450980394

performance (no tuning )
case: SLAB
rfc test accuracy = 0.624
case: WET
rfc test accuracy = 0.567

__type imbalance:__ 510 days, 361 slab, 71 wet. 141 predictions after ttsplit

predicting on correct SLAB/WET column:
case: SLAB
rfc test accuracy = 0.837
case: WET
rfc test accuracy = 0.851

performance at p = 0.5
slab
acc: 0.8368794326241135
prec: 0.8455284552845529
rec: 0.9629629629629629

wet
acc: 0.851063829787234
prec: 0.5
rec: 0.19047619047619047

__predictions are imbalanced in the other direction:__
slab: 108/141, 0.7659574468085106

wet: 21/141, 0.14893617021276595

days with either: 129/141, 0.9148936170212766 (only 7 with both)

### combined model: pretty good! better with t=0.4 (min=0.34)
slab
acc: 0.8368794326241135
prec: 0.8455284552845529
rec: 0.9629629629629629

wet
acc: 0.851063829787234
prec: 0.5
rec: 0.19047619047619047

sum
acc: 0.7872340425531915
prec: 0.8709677419354839
rec: 0.8852459016393442

In [120]: roc.confusion_mtx()
Out[120]: (108, 3, 16, 14)

### NEED MORE non-avy days in test set. Naive model (1s):
__where are days being removed? at the 'zone' step in feature_engineering__
In [124]: roc.fit(sum_true, y_naive)

In [125]: roc.accuracy
Out[125]: 0.81560283687943258

In [126]: roc.precision
Out[126]: 0.85820895522388063

In [127]: roc.recall
Out[127]: 0.94262295081967218

### better: left merge (weather, zone_df) to save days w/o avalanche
__test class balance:__ 32.6% positive class (90/276)

results: no class balancing:
slab
acc: 0.9492753623188406
prec: 0.8846153846153846
rec: 0.9324324324324325

wet
acc: 0.9384057971014492
prec: 0.42857142857142855
rec: 0.1875

sum
acc: 0.9094202898550725
prec: 0.8734177215189873
rec: 0.8214285714285714

naive: ones
In [44]: roc.accuracy
Out[44]: 0.28260869565217389

In [45]: roc.precision
Out[45]: 0.28888888888888886

In [46]: roc.recall
Out[46]: 0.9285714285714286

naive: zeros
In [51]: roc.accuracy
Out[51]: 0.71739130434782605

In [52]: roc.precision
Out[52]: 1.0

In [53]: roc.recall
Out[53]: 0.071428571428571425
