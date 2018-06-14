### Avalanche prediction processing pipeline

### data: .csv files in /data/

### data cleaning
  __clean_data_caic.py__
   - input: .csv
   - output: sqlite database (data-caic.db)
   - actions: read avy data, make format corrections, save as .csv files
   - script dependencies:
      - cleaning_scripts.py, sqlite3_scripts.py

  __clean_data_bczone.py__
   - input: .csv
   - output: sqlite database (data-aspen.db, data-nsj.db)
   - actions: read snotel and wind data for a backcountry zone, make format corrections, save as .csv files
   - script dependencies:
      - cleaning_scripts.py, sqlite3_scripts.py

### feature engineering
  __feature_engineering.py__
   - input: sqlite database for zone
   - output: sqlite database (data-engineered.db), one table for each bc zone
   - actions:
     - engineer features
     - convert dates to water year
     - engineer timeseries lag features
     - impute NaNs (with mean or other value)
   - script dependencies:
      - transformation_scripts.py, sqlite3_scripts.py

### modeling
  __train_classifier.py__ (or train_classifier_rfc.py)
   - input: sqlite database
     - cleaned and engineered feature matrix as pandas df
   - output: saved as .pkl in /best-ests/
     - fitted estimators and standardizers
     - saves one set for each of two cases: 'slab' and 'wet'
   - script dependencies:
      - transformation_scripts.py, modeling_scripts.py, sqlite3_scripts.py

  __predict_fitted_classifier.py__
   - input: sqlite database, .pkl files
     - cleaned and engineered feature matrix as pandas df
     - fitted estimator and standardizer
   - output: saved as .pkl in /outputs/
     - predicted binary and predicted probability for both cases
     - feature names and importances

  __output_classifier.py__
   - input: .pkl from /outputs/
     - outputs from predictions
   - output:
     - figures

### ancillary: grid search scripts
  __train_classifier_gridsearch_gbc.py__ (or rfc version)
   - train model with large grid of parameters using SkLearn GridSearchCV

### ancillary: eda, visualization, etc
  - EDA scripts in src/eda/
  - kde_probabilities.py
    - models probability of slab/wet avalanche as a Gaussian KDE
    - script dependencies
      - transformation_scripts.py
  - ts_results_plot.py
    - makes a timeseries plot of results: actual and predicted

__dataframe sizes:__

clean_data outputs:
 - avy_df: (10151, 40)
 - snotel_df: (74532, 10), without airtemp cleaning: (74636, 10)
 - airport_df: (7474,4)


feature_engineering
 - inputs:
    - avy_df: (10128, 43)
    - snotel_df: (43266, 12)
    - airport_df: (7474, 4)

    - zone_df:  (510, 9), float and int
 - outputs:
    -merge_all: (510, 24), float and int

__directory structure:__
~~~
├── README.md
├── ROC.py
├── best-ests
│   ├── aspen_best_est_gbc_SLAB.p
│   └── aspen_best_est_gbc_WET.p
│
├── clean_data_bczone.py
├── clean_data_caic.py
├── cleaning_scripts.py
├── eda
│   ├── eda_avy.py
│   ├── eda_data.py
│   └── eda_snow.py
│
├── feature_diagnostics.py
├── feature_engineering.py
├── feature_hists_aspen
│   ├── histograms (.png)
│
├── feature_hists_nsj
│   ├── histograms (.png)
│
├── kde_probabilities.py
├── model_features.md
├── modeling_notes.md
├── modeling_scripts.py
├── output_classifier.py
├── outputs
│   └── aspen_gbc_output.p
│
├── plotting_scripts.py
├── predict_fitted_classifier.py
├── predict_partial_dependence.py
├── sqlite3_scripts.py
├── train_classifier.py
├── train_classifier_gridsearch_gbc.py
├── train_classifier_gridsearch_rfc.py
├── transformation_scripts.py
└── ts_results_plot.py
~~~
