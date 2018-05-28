### src folder readme

### Avalanche prediction processing pipeline

#### data: .csv files in /data/

#### data cleaning
  __clean_data.py__
   - in: .csv
   - out: .csv
   - scripts:
      - cleaning_scripts.py

#### feature engineering
  __feature_engineering.py__
   - in: .csv
   - out: .pkl
   - actions: engineer features, engineer ts lag features,
   - scripts:
      - transformation_scripts.py

#### modeling
   - run_model_2options.py

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

dataframe sizes:

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

 - nans:
N_AVY                   0
MONTH                   0
DOY                     0
N_AVY_24                0
DSUM_24                 0
P_SLAB                  0
P_WET                   0
WET                     0
SLAB                    0
WSP_SSTN_aspen         64
WSP_PEAK_aspen         65
WSP_SSTN_leadville     76
WSP_PEAK_leadville     91
DEPTH                  46
GRTR_60                46
SNOW_24                46
SNOW_4DAY              46
SWE_24                 46
DENSE_24              306
SETTLE                306

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
