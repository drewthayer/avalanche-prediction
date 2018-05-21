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
