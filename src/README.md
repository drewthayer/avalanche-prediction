### Avalanche prediction processing pipeline

#### data: .csv files in /data/

#### data cleaning
  __clean_data.py__
   - input: .csv
   - output: .csv
   - actions: read data, make format corrections, save as .csv files
   - scripts:
      - cleaning_scripts.py

#### feature engineering
  __feature_engineering.py__
   - input: .csv
   - output: .pkl
   - actions:
     - engineer features
     - convert dates to water year
     - engineer timeseries lag features
     - impute NaNs (with mean or other value)
   - scripts:
      - transformation_scripts.py

#### modeling
  __train_classifier_gbc.py__ (or train_classifier_rfc.py)
   - input: (.pkl)
     - cleaned and engineered feature matrix as pandas df
   - output:
     - fitted estimator and standardizer, as pickle
     - saves one set for each of two cases: 'slab' and 'wet'
  __predict_fitted_classifier.py__
   - input: (.pkl)
     - cleaned and engineered feature matrix as pandas df
     - fitted estimator and standardizer
   - output:
     - predicted binary and predicted probability for both cases
     - feature names and importances
  __output_classifier.py__
   - input: (.pkl)
     - outputs from predictions
   - output:
     - figures 
~~~
/project
  /data
      .csv files
  /src
    /pkl
        .pkl files
    data prep:
    clean_data.py
        functions:

        scripts:
            cleaning_scripts.py_
                clean_airport_data.py (removes hourly data: only rows with DAILY columns )
                clean_snow_data.py
                remove_airtemp_outliers.py
    feature_engineering.py_
        transformation_scripts.py
            water_year_day.py
            water_year_month.py
            oversample.py

    modeling: (DROPNA HERE )
    run_model.py
    run_model_slab_wet.py
    run_model_2options.py

    model outputs:
    output.py

~~~

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
├── ROC.py
├── best-ests
│   ├── pickled trained estimators (.pkl)
├── clean_data.py
├── cleaning_scripts.py
├── eda
│   ├── Aspen_navy_ts.png
│   ├── eda_avy.py
│   ├── eda_data.py
│   └── eda_snow.py
├── feature_diagnostics.py
├── feature_engineering.py
├── feature_hists
│   ├── histograms for aspen area (.png)
├── feature_hists_nsj
│   ├── histograms for nsj area (.png)
├── kde_probabilities.py
├── main.py
├── modeling_scripts.py
├── output.py
├── output_classifier.py
├── pkl
│   ├── pickle files (.p)
├── plotting_scripts.py
├── predict_fitted_classifier.py
├── predict_partial_dependence.py
├── run_2models_classification.py
├── train_classifier.py
├── train_classifier_gbc.py
├── train_classifier_gridsearch_gbc.py
├── train_classifier_gridsearch_rfc.py
├── transformation_scripts.py
└── ts_results_plot.py
~~~
