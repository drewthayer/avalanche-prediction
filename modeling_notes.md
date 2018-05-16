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
