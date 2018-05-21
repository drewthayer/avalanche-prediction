Readme
## Empirical avalanche prediction in Colorado:
#### Can a machine-learning model trained on historical climatic and avalanche data augment prediction of avalanche risk?

__A capstone project for the Galvanize Data Science Immersive.__

<img alt="avy" src="/figs/pub_figs/skier_avy.png" width='500'>

_source: Staying Alive in Avalanche Terrain, Bruce Tremper, The Mountaineers Books_

__DISCLAIMER:__ This information is NOT intended to be used as an avalanche risk forecast. This is an empirical study done for scientific purposes. Refer to the professionals for avalache forecasts:

http://avalanche.state.co.us

## Preliminary work (as of April 24, 2018)

### Data:
__Colorado Avalanche Information Center data__

(Colorado Department of Natural Resources)

10 backcountry zones:

<img alt="caic zones" src="/figs/pub_figs/CAIC_zones.png" width='300'>

avalanche observation data back to 1980:

<img alt="caic zones" src="/figs/pub_figs/caic_example.png" width='500'>

__weather data__
SNOTEL sensor network (NRCS, USDA):

<img alt="snotel network" src="/figs/pub_figs/co_swe_current.png" width='500'>

<img alt="snotel network" src="/figs/pub_figs/nrcs_snotel_eyak_ak.jpg" width='200'>

_source: NRCS National Water and Climate Center, USDA_

Local Climatalogical Data (commonly airports):

<img alt="airport station" src="/figs/pub_figs/airport_weather_station.JPG" width='200'>

### avalanche trends:
__destructive size:__

<img alt="avy by location" src="/figs/eda/dsize.png" width='300'>

_this modeling approach will consider avalanches D2 or greater_

__D2+ avalanches by backcountry zone:__
- Northern San Juan        2998
- Front Range              1565
- Vail & Summit County     1337
- Aspen                    1210
- Gunnison                 1188
- Sawatch Range             806
- Southern San Juan         585
- Steamboat & Flat Tops     186
- Grand Mesa                155
- Sangre de Cristo           22


### modeling strategy:

__preliminary study: Aspen zone__

<img alt="caic zones" src="/figs/pub_figs/aspen_closeup.png" width='200'>

__Data:__  
 - _features:_ wind data from Aspen and Leadville airports, air temperature and precipitation data from Independence Pass SNOTEL station
 - _target:_ Aspen Zone avalanches, # per day (size >= D2)
 - _train and test split:_ June 2016

<img alt='timeseries' src='figs/eda/aspen_avys_d2plus.png' width='500'>

### current state:
__best model: random forest classifier__
 - test accuracy = 0.942
 - test recall = 0.932

### feature engineering
 - use features that control the physical processes that create avalanche conditions:
   - snowfall, wind, temperature
 - literature: features used in avalanche modeling study in Little Cottonwood Canyon, UT
   - _Blatternberger and Fowles, 2016. Treed Avalanche Forecasting: Mitigating Avalanche Danger Utilizing Bayesian Additive Regression Trees. Journal of Forecasting, J. Forecast. 36, 165–180 (2017). DOI: 10.1002/for.2421_

 __probabilities of slab/ wet avalanches:__
  - slab and wet avalanches have overlapping, yet different seasons:
<img alt='timeseries' src='figs/eda/types_by_month_slab_WL.png' width='500'>

  - relative probability modeled as Gaussian KDE function:
<img alt='timeseries' src='figs/eda/types_by_day_nonorm.png' width='500'>

  - each day, calculate p(slab) and p(wet) as function of day-of-water-year

### modeling:
__training:__
 - training data: Aspen Zone, period of record from 2010-2016, CAIC records for D rating > 2 avalanches
 - target: binary label for occurence of any avalanches with D rating > 2
 - separate targets for slab and wet occurence
 - separate models for slab and wet avalanches
   - each model does not include # or probability of other type of avalanche

__testing:__
- test data: 2016 - 2018 period of record
- predict probability of positive class with each model (slab/wet avalanche occured)
- add probabilities: $\Sigma p_{slab}, p_{wet} = p_{avalanche}$
- evaluate model accuracy and recall  
