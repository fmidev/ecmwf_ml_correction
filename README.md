# Machine learning (XGB) based bias correction for ecmwf temperature and dewpoint deterministic weather forecasts
This Machine Learning (ML) correction code can be used to correct the forecast errors in deterministic weather model forecasts for 1-240 hour leadtimes. The input data is the ecmwf model grids (grib2-files) and the output is corrected model fields (grib2). Machine learning correction is an eXtreme Gradient Boosting (XGBoost) regressor based on 5 years of training data and is calculated for points which are then further gridded back to NWP model background using Gridpp. ML correction is available for following parameters: 2m temperature and 2m dewpoint temperature. 

For 2m temperature, the prediction code makes also min/max temperature combination for the first 5 days forecasts. Which means that:
- when 2m temperature is lowest at night time (19-6 UTC), the 2m temperature forecast is replaced with 12 hours minimum temperature forecast
- when 2m temperature is highest at day time (7-18 UTC), the 2m temperature forecast is replaced with 12 hours maximum temperature forecasts

Predction code also checks that:
 - 2m dewpoint temperature forecast can not be higher than 2m temperature forecast at any grid point

## Usage
Running with run_predict.sh shell script:
```
./run_predict.sh YYYYMMDDHH parameter outfile.grib2 producer_id 
```
E.g. 
```
./run_predict.sh 2026090800 "temperature" "temperature_2026090800.grib2" 122
```

## Authors
kaisa.ylinen@fmi.fi
