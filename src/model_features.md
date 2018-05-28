__Table I. Variables used in the analysis__

| feature | source | imputation |feature (lit) | description |
|----|-----|-----|----|----|
| YEAR | _CAIC_ | | YEAR | |
| MONTH | _CAIC_ | | MONTH | |
| DAY | _CAIC_ |  | DAY | |
| n_avy | _CAIC_ | | AVAL 0–1 | Avalanche crosses road |
| na | na | | CLOSE 0–1 | Road closed |
| snow_h | _SNOTEL: precip_start_m_ | |TOTSTK Total stake: total snow | depth in inches|
| GRTR_40 | _SNOTEL_ | 0 (marginal)|TOTSTK60 | If TOTSTK>60 cm. TOTSTK60 = TOTSTK60 (cm) |
| snow_last_24 | _SNOTEL_ | 0 |  INTSTK | Interval stake: depth of snowfall in last 24 hours |
| w_4day_snow | _SNOTEL_ | | SUMINT | Weighted sum of snow fall in last 4 days: weights = (1.0, 0.75, 0.50, 0.25) |
| snow_density | _SNOTEL_ | DENSITY | Density of new snow, ratio of water content of new snow to new snow depth |
| rel_density | _SNOTEL_ | RELDEN | Relative density of new snow, ratio of density of new snow to density of previous storm |
| t_max_sum | _SNOTEL_ |SWARM |Sum of maximum temperature on last three ski days, an indicator of a warm spell |
| settle | _SNOTEL_ | SETTLE | Change in TOTSTK60 relative to depth of snowfall in the last 24 hours |
| swe | _SNOTEL_ | WATER | Water content of new snow measured in mm |
| t_min_delta | _SNOTEL_ | CHTEMP | Difference in minimum temperature from previous day |
| t_min_24 | _SNOTEL_ | TMIN | Minimum temperature in last 24 hours |
| t_max_24 | _SNOTEL_ | TMAX | Maximum temperature in last 24 hours |
| wsp_max | _airport_ | WSPD | Wind speed, mph at peak location |
| wsp_sustained | _airport_ | | |
| avy_24_n | _CAIC_ | NAVALLAG | Number of avalanches crossing the road on previous day |
| avy_24_dsum | _CAIC_ | SZAVLAG | Size of avalanche, sum of size ratings for all avalanches in NAVALLAG |

other: features based on storms

 - STMSTK Storm stake: depth of new snow in previous storm

other: dont apply to this model
 - HAZARD Hazard rating of avalanche forecasters
 - NART Number of artificial explosives used
