#helper functions for ecmwf ml correction prediction

import os
import eccodes as ecc
import gridpp
import numpy as np
import sys
import datetime
import pandas as pd
import math
import xgboost as xgb
import fsspec
from multiprocessing import Process, Queue

def read_file_from_s3(grib_file):
    uri = "simplecache::{}".format(grib_file)
    
    return fsspec.open_local(
        uri,
        mode="rb",
        s3={"anon": True, "client_kwargs": {"endpoint_url": "https://eris.fmi.fi"}},
    )


def read_grib(gribfile, read_coordinates=False):
    """Read first message from grib file and return content.
    List of coordinates is only returned on request, as it's quite
    slow to generate.
    """
    forecasttime = []
    leadtime = []
    values = []
    
    print(f"Reading file {gribfile}")
    wrk_gribfile = gribfile
    
    if gribfile.startswith("s3://"):
        wrk_gribfile = read_file_from_s3(gribfile)
        
    lons = []
    lats = []
    
    with open(wrk_gribfile, "rb") as fp:
        # print("Reading {}".format(gribfile))
        
        while True:
            try:
                gh = ecc.codes_grib_new_from_file(fp)
            except ecc.WrongLengthError as e:
                print(e)
                file_stats = os.stat(wrk_gribfile)
                print("Size of {}: {}".format(wrk_gribfile, file_stats.st_size))
                sys.exit(1)
            
            if gh is None:
                break
            
            ni = ecc.codes_get_long(gh, "Nx")
            nj = ecc.codes_get_long(gh, "Ny")
            dataDate = ecc.codes_get_long(gh, "dataDate")
            dataTime = ecc.codes_get_long(gh, "dataTime")
            forecastTime = ecc.codes_get_long(gh, "endStep")
            leadtime.append(forecastTime)
            analysistime = datetime.datetime.strptime(
                "{}.{:04d}".format(dataDate, dataTime), "%Y%m%d.%H%M"
            )
            
            ftime = analysistime + datetime.timedelta(hours=forecastTime)
            forecasttime.append(ftime)
            
            tempvals = ecc.codes_get_values(gh).reshape(nj, ni)
            values.append(tempvals)

            if read_coordinates and len(lons) == 0:
                lats = np.asarray(ecc.codes_get_array(gh, "latitudes"),dtype="float64").reshape(nj, ni)
                lons = np.asarray(ecc.codes_get_array(gh, "longitudes"),dtype="float64").reshape(nj, ni)
            ecc.codes_release(gh)
        

        #Order values, leadtime, forecasttime based on leadtime
        sorted_indices = np.argsort(leadtime)
        values = [values[i] for i in np.array(sorted_indices)]
        leadtime.sort()
        forecasttime.sort()
                                                        
        if read_coordinates == False and len(values) == 1:
            return (
                None,
                None,
                np.asarray(values).reshape(nj, ni),
                leadtime,
                analysistime,
                forecasttime,
            )
        elif read_coordinates == False and len(values) > 1:
            return None, None, np.asarray(values), leadtime, analysistime, forecasttime
        else:
            return (
                lons,
                lats,
                np.asarray(values),
                leadtime,
                analysistime,
                forecasttime,
            )


def read_grid(args, variable):
    """Top function to read "all" gridded data"""
    # Define the grib-file used as background/"parameter_data"
    if variable == "temperature":
        lons, lats, vals, leadtime, analysistime, forecasttime = read_grib(args.t2_data, True)
    elif variable == "dewpoint":
        lons, lats, vals, leadtime, analysistime, forecasttime = read_grib(args.td2_data, True)
    elif variable == "t_max":
        lons, lats, vals_1h, leadtime_1h, analysistime, forecasttime_1h = read_grib(args.tmax_data, True)
        _, _, vals_3h, leadtime_3h, _, forecasttime_3h = read_grib(args.tmax3h_data, False)

    elif variable == "t_min":
        lons, lats, vals, leadtime, analysistime, forecasttime = read_grib(args.tmin_data, True)

    _, _, topo, _, _, _ = read_grib(args.topography_data, False)
    _, _, lc, _, _, _ = read_grib(args.landseacover_data, False)

    # modify  geopotential to height and use just the first grib message, since the topo & lc fields are static
    topo = topo / 9.81
    if(len(topo.shape) == 3): topo = topo[0]
    if(len(lc.shape) == 3): lc = lc[0]

    grid = gridpp.Grid(lats, lons, topo, lc)
    return grid, lons, lats, vals, leadtime, analysistime, forecasttime, lc, topo

def interpolate_to_point(field, i, j, wx, wy):
    '''Bilinear interpolation of gridded field (time,j,i) to a point defined by i,j and weights wx, wy.'''
    return (
        (1 - wx) * (1 - wy) * field[:, j,     i]
        + wx     * (1 - wy) * field[:, j,     i + 1]
        + (1 - wx) * wy     * field[:, j + 1, i]
        + wx     * wy       * field[:, j + 1, i + 1]
    )

def create_features_data(args):
    '''Create features array which has dimensions for times, stations, variables.
    In addition create metadata that contains (valid)time and leadtime.''' 
    #Station list 
    stations = pd.read_csv(args.stations_list)

    #Retrieve data in loop to save memory, fetch only one field to python at once
    features_list  = ["p","t2m","td2m","u10m","v10m","lcc","mcc","skt","mx2t","mn2t","t925","t_ensmean"]
    _, _, data, leadtime, _, forecasttime = read_grib(args.p_data, False)
    time_values = [ x.strftime('%Y-%m-%d %H:%M:%S') for x in forecasttime]
    metadata = pd.DataFrame(data = {'leadtime': leadtime,'time': time_values})
    #Take four closest grid points and save them to features array
    point_values = interpolate_to_point(data, stations["i"].to_numpy(), stations["j"].to_numpy(), stations["wx"].to_numpy(), stations["wy"].to_numpy())
    features = np.empty((len(data), len(stations), 12)) #third column is for parameters (Obs! p is read outside the following loop)
    features[:] = np.nan
    features[:,:,0] = point_values
    del data, leadtime, forecasttime
    i = 1
    for param_args in [args.t2_data,args.td2_data,args.u10_data,args.v10_data,args.lcc_data,args.mcc_data,
                       args.skt_data,args.tmax_data,args.tmin_data,args.t925_data,args.t_ensmean_data]:
        _, _, data, _, _, _ = read_grib(param_args, False)
        #Take four closest grid points
        point_values = interpolate_to_point(data, stations["i"].to_numpy(), stations["j"].to_numpy(), stations["wx"].to_numpy(), stations["wy"].to_numpy())
        if len(point_values) == 65:  # ensmean has 3h time resolution for the first 90 hours
            features[:91:3,:,i] = point_values[0:31]
            features[91:,:,i] = point_values[31:65]
        elif len(point_values) == 124:  # tmin and tmax starts from leadtime 1
            features[1:,:,i] = point_values
        else:
            features[:,:,i] = point_values
        del data
        i += 1
        
    #Add Time lagged features T2_M1 and T_925_M1 (leadtime 0h will be discarded)
    T2_M1 = features[:-1,:,features_list.index("t2m")]
    T_925_M1 = features[:-1,:,features_list.index("t925")]
    
    #Add lagged features to features array and reshape to two dimensional array (times*stations, features)
    features2 = np.concatenate((features[1:,:,:], T2_M1[:,:,None], T_925_M1[:,:,None]), axis=2).reshape(-1,features.shape[2]+2)
    features_list = features_list + ["t2m_M1", "t925_M1"]
        
    #Create time features
    datetime_object = pd.to_datetime(metadata['time'], format = '%Y-%m-%d %H:%M:%S')
    is_leap_year = datetime_object.dt.is_leap_year.fillna(False).to_numpy(dtype=bool)
    days_in_year = np.where(is_leap_year, 366, 365)
    features_time = pd.DataFrame({
        'leadtime': np.array(metadata['leadtime']),
        'hod_sin': np.sin(datetime_object.dt.hour*2*math.pi/24),
        'hod_cos': np.cos(datetime_object.dt.hour*2*math.pi/24),
        'doy_sin': np.sin((datetime_object.dt.dayofyear-1)*2*math.pi/days_in_year),
        'doy_cos': np.cos((datetime_object.dt.dayofyear-1)*2*math.pi/days_in_year),
        'analysishour': np.repeat(int(args.analysis_time[-2:]),len(metadata))
        })
    time_features = np.repeat(np.array(features_time)[metadata['leadtime']>0],len(stations),axis=0)
    
    #Create station features
    features_station = pd.DataFrame({
        'lon': stations['lon'],
        'lat': stations['lat'],
        'elev': stations['elev']})
    station_features = np.repeat(np.array(features_station)[None,:,:],len(metadata['leadtime'])-1, axis=0).reshape(-1,3)
    
    #Combine time and station features and reorder them to match order used in training
    meta_features = np.concatenate((time_features[:,0][:,None], station_features, time_features[:,1:]), axis=1)
    features_list = features_list + ['leadtime', 'lon', 'lat', 'elev', 'hod_sin', 'hod_cos', 'doy_sin', 'doy_cos', 'analysishour']
    
    #Combine all features to one two dimensional array
    all_features = np.concatenate((features2, meta_features),axis=1)
    
    return all_features, features_list


def get_point_forecasts(variable, all_features, features_list):
    '''Give point forecasts for the specified variable'''
    #Select forecasts_point for given variable for error calculation
    if (variable == "temperature"): iloc = [features_list.index("t2m")]
    if (variable == "dewpoint"): iloc = [features_list.index("td2m")]
    if (variable == "t_max"): iloc = [features_list.index("mx2t")]
    if (variable == "t_min"): iloc = [features_list.index("mn2t")]
    forecasts_point = all_features[:,iloc[0]].reshape(-1)  

    return forecasts_point


def xgb_prediction(all_features, forecasts_point, args, variable):
    '''Make xgb prediction for given variable'''
        
    #Load xgb model
    xgb_model = xgb.XGBRegressor()
    if (variable == "temperature"): xgb_model.load_model(args.model_ta)
    if (variable == "dewpoint"): xgb_model.load_model(args.model_td)
    if (variable == "t_max"): xgb_model.load_model(args.model_tmax)
    if (variable == "t_min"): xgb_model.load_model(args.model_tmin)

    #Predict forecast correction and calculate forecast values for quantile mapping
    xgb_predict = xgb_model.predict(all_features)
    xgb_forecast = forecasts_point + xgb_predict #XGB model predicts the forecast error (obs - forecast)

    return xgb_forecast


def select_indices(ahour):
    '''Select indeces for tmin and tmax combining based on hour of analysis time.
    Indeces tells when time is between 7-18 UTC for indeces_max_first/last and 19-6 UTC for indeces_min_first/last.
    Leadtimes for ml corrected ecmwf are 1...240h, but min/max correction is made only for the first 5 days'''
    if (ahour == 0):
        indices_max_first = [6, 30, 54, 78, 94, 102]
        indices_max_last = [18, 42, 66, 90, 98, 106]
        indices_min_first = [18, 42, 66, 90, 98]
        indices_min_last = [30, 54, 78, 94, 102]
    elif (ahour == 12):
        indices_max_first = [18, 42, 66, 90, 98]
        indices_max_last = [30, 54, 78, 94, 102]
        indices_min_first = [6, 30, 54, 78, 94, 102]
        indices_min_last = [18, 42, 66, 90, 98, 106]
    return indices_max_first, indices_max_last, indices_min_first, indices_min_last


def ml_predict(args, all_features, features_list, variable):
    '''Calculate ml_correction and return predicted corrections in list where 
    each element is for one lead time'''
    forecasts_point = get_point_forecasts(variable, all_features, features_list)

    if variable == "temperature": #Make min/max combining for temperature forecasts
        #Create features for xgb model        
        forecasts_point_tmax = get_point_forecasts("t_max", all_features, features_list)
        forecasts_point_tmin = get_point_forecasts("t_min", all_features, features_list)
        
        #Make xgb prediction
        xgb_forecast_ta = xgb_prediction(all_features, forecasts_point, args, variable)
        xgb_forecast_tmax = xgb_prediction(all_features, forecasts_point_tmax, args, "t_max")
        xgb_forecast_tmin = xgb_prediction(all_features, forecasts_point_tmin, args, "t_min")
        
        ##############################################################
        #Combine min/max forecasts to temperature (only first 5 days)#
        ##############################################################
        xgb_forecast_with_min_max = xgb_forecast_ta.copy()
        
        #Chech number of stations
        all_stations = pd.read_csv(args.stations_list)
        stations_n = len(all_stations['SID'])

        #Set thresholds how much tmax and tmin can differ from t2m
        threshold_tmax = 2.5 
        threshold_tmin = 2 

        #Check month of analysis time and set thresholds for autumn/winter time
        amonth = int(args.analysis_time[4:6])
        if (amonth >= 10) | (amonth <= 2): #Months Oct...Feb
            threshold_tmin = 1
            threshold_tmax = 2

        #Check hour of analysis time and select indices based on them 
        ahour = int(args.analysis_time[-2:])
        indices_max_first, indices_max_last, indices_min_first, indices_min_last = select_indices(ahour)

        #Loop over all stations
        for i in range(stations_n):
            xgb_forecast_station = xgb_forecast_ta[i::stations_n].copy()
            xgb_forecast_tmax_station = xgb_forecast_tmax[i::stations_n]
            xgb_forecast_tmin_station = xgb_forecast_tmin[i::stations_n]
            
            #Loop separately over day (indices_max) and night (indices_min) times
            for j0,j1 in zip(indices_max_first, indices_max_last):
                #Select all leadtimes for one day (7-18 UTC). Calculate tmax forecast, maximum of temperature forecasts and location of maximum temperature 
                tmax_forecast = np.mean(xgb_forecast_tmax_station[j0:j1])
                temperature_max = np.max(xgb_forecast_station[j0:j1])
                max_indices = np.where(xgb_forecast_station[j0:j1] == temperature_max)[0]
                iloc_temperature_max = max_indices[len(max_indices)//2] #select the middle index if multiple times have max temperature
                #If tmax forecast is higher than maximum temperature, set tmax forecast as new maximum temperature
                #Set a limit, that a new maximum temperature can be only amount of threshold higher than maximum temperature
                if (tmax_forecast > temperature_max):
                    tmax_limited = min((tmax_forecast - temperature_max), threshold_tmax)
                    xgb_forecast_station[j0:j1][iloc_temperature_max] += tmax_limited  
            for j0,j1 in zip(indices_min_first, indices_min_last):
                #Select all leadtimes for one night (19-6 UTC). Calculate tmin forecast, minimum of temperature forecasts and location of minimum temperature 
                tmin_forecast = np.mean(xgb_forecast_tmin_station[j0:j1])
                temperature_min = np.min(xgb_forecast_station[j0:j1])
                min_indices = np.where(xgb_forecast_station[j0:j1] == temperature_min)[0]
                iloc_temperature_min = min_indices[len(min_indices)//2] #select the middle index if multiple times have min temperature
                #If tmin forecast is lower than minimum temperature, set tmin forecast as new minimum temperature
                #Set a limit, that a new minimum temperature can be only amount of threshold lower than minimum temperature
                if (tmin_forecast < temperature_min):
                    tmin_limited = max(tmin_forecast - temperature_min, -threshold_tmin)
                    xgb_forecast_station[j0:j1][iloc_temperature_min] += tmin_limited
            #Write min/max corrected station values to xgb_forecast_with_min_max array
            xgb_forecast_with_min_max[i::stations_n] = xgb_forecast_station
        xgb_forecast = xgb_forecast_with_min_max
    else:
        xgb_forecast = xgb_prediction(all_features, forecasts_point, args, variable)
    
    #Predictions back to forecast corrections
    ml_correction = forecasts_point - xgb_forecast ###Pohdi vielä kummin päin erotus on järkevämpi tehdä!

    #Set limit for ml_correction values that they can be only between -10...8
    ml_correction[ml_correction > 8] = 8
    ml_correction[ml_correction < -10] = -10  

    # Store data to list where each leadtime is own item (leadtimes: +1h..+240h)
    ml_results = []
    leadtimes = all_features[:, features_list.index('leadtime')] #leadtime column

    for j in np.unique(leadtimes):
        ml_results.append(ml_correction[leadtimes == j])

    return ml_results


def get_points(args):
    '''Create point variable for gridpp interpolation'''
    all_stations = pd.read_csv(args.stations_list)
    
    points = gridpp.Points(
        all_stations['lat'].to_numpy(),
        all_stations['lon'].to_numpy(),
        all_stations['elev'].to_numpy(),
        #all_stations['lsm'].to_numpy(),
    )

    return points


def interpolate_single_time(
            grid,
            background,
            points,
            obs,
            obs_to_background_variance_ratio,
            pobs,
            structure,
            max_points,
            idx,
            q,
):
    # perform optimal interpolation
    tmp_output = gridpp.optimal_interpolation(
        grid,
        background,
        points,
        obs[idx],
        obs_to_background_variance_ratio,
        pobs,
        structure,
        max_points,
    )

    print(
        "step {} min grid: {:.1f} max grid: {:.1f}".format(
            idx, np.amin(tmp_output), np.amax(tmp_output)
        )
    )

    if q is not None:
        # return index and output, so that the results can
        # later be sorted correctly
        q.put((idx, tmp_output))
    else:
        return tmp_output


def interpolate(grid, points, background, obs, args, lc):
    """Perform optimal interpolation"""
    
    output = []
    
    # create a mask to restrict the modifications only to land area (where lc = 1)
    lc0 = np.logical_not(lc).astype(int)
    
    # Interpolate background data to observation points
    # When bias is gridded then background is zero so pobs is just array of zeros
    pobs = gridpp.nearest(grid, points, background)
    
    # Barnes structure function with horizontal decorrelation length 30km,
    # vertical decorrelation length 200m
    structure = gridpp.BarnesStructure(30000, 200, 0.5)
    
    # Include at most this many "observation points" when interpolating to a grid point
    max_points = 20
    
    # error variance ratio between observations and background
    # smaller values -> more trust to observations
    obs_to_background_variance_ratio = np.full(points.size(), 0.1)
    
    if args.disable_multiprocessing:
        output = [
            interpolate_single_time(
                grid,
                background,
                points,
                obs,
                obs_to_background_variance_ratio,
                pobs,
                structure,
                max_points,
                x,
                None,
            )
            for x in range(len(obs))
        ]
        
    else:
        q = Queue()
        processes = []
        outputd = {}
        
        for i in range(len(obs)):
            processes.append(
                Process(
                    target=interpolate_single_time,
                    args=(
                        grid,
                        background,
                        points,
                        obs,
                        obs_to_background_variance_ratio,
                        pobs,
                        structure,
                        max_points,
                        i,
                        q,
                    ),
                )
            )
            processes[-1].start()
            
        for p in processes:
            # get return values from queue
            # they might be in any order (non-consecutive)
            ret = q.get()
            outputd[ret[0]] = ret[1]
            
        for p in processes:
            p.join()
            
        for i in range(len(obs)):
            # sort return values from 0 to 8
            output.append(outputd[i])
            
    return output


def ml_corrected_forecasts(forecasttime, background, diff, variable):
    '''calculate the final ml corrected forecast fields: MEPS - ml_correction
    and make rough qc to forecasts'''
    # Remove leadtimes 0, because due to lagged features, correction is not made to those
    n_lags = len(forecasttime) - len(diff)
    output = []
    for j in range(0, len(diff)):
        tmp_output = background[j + n_lags] - diff[j]
        # Implement simple QC thresholds
        if variable == "temperature":
            tmp_output = np.clip(tmp_output, 218, 328)
        elif variable == "dewpoint":
            tmp_output = np.clip(tmp_output, 208, 323)
        output.append(tmp_output)

    forecasttime = forecasttime[n_lags:]
    assert len(forecasttime) == len(output)
    
    return output, forecasttime


def write_grib_message(fp, args, analysistime, forecasttime, data):
    pdtn = 70
    tosp = None
    if args.parameter == "temperature":
        pnum = 0
        pcat = 0
        levelvalue = 2
    elif args.parameter == "dewpoint":
        pnum = 6
        pcat = 0
        levelvalue = 2
    elif args.parameter == "t_max": #??
        pdtn = 11
        pnum = 0
        pcat = 0
        levelvalue = 2
        tosp = 2
    elif args.parameter == "t_min": #??
        pdtn = 11
        pnum = 0 
        pcat = 0
        levelvalue = 2
        tosp = 2
    # Store different time steps as grib msgs
    forecastTime_prev = 0
    for j in range(0, len(data)):
        tdata = data[j]
        forecastTime = int((forecasttime[j] - analysistime).total_seconds() / 3600)
        
        # - For non-aggregated parameters, grib2 key 'forecastTime' is the time of the forecast
        # - For aggregated parameters, it is the start time of the aggregation period. The end of the period is defined by 'lengthOfTimeRange'
        #   ECMWF have different time steps for different lead times: 1-90 (1h time interval), 93-144 (3h time interval), 150-240 (6h time interval),
        #   so save previous forecastTime end of this loop iteration
        iouot = forecastTime - forecastTime_prev

        if tosp == 2:
            forecastTime = forecastTime_prev
        
        h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        ecc.codes_set(h, "tablesVersion", 28)
        ecc.codes_set(h, "shapeOfTheEarth", 0)
        ecc.codes_set(h, "Ni", tdata.shape[1])
        ecc.codes_set(h, "Nj", tdata.shape[0])
        ecc.codes_set(h, "latitudeOfFirstGridPointInDegrees", 73.5)
        ecc.codes_set(h, "longitudeOfFirstGridPointInDegrees", -40.0)
        ecc.codes_set(h, "latitudeOfLastGridPointInDegrees", 27.5)
        ecc.codes_set(h, "longitudeOfLastGridPointInDegrees", 72.5)
        ecc.codes_set(h, "jDirectionIncrementInDegrees", 0.1)
        ecc.codes_set(h, "iDirectionIncrementInDegrees", 0.1)
        ecc.codes_set(h, "dataDate", int(analysistime.strftime("%Y%m%d")))
        ecc.codes_set(h, "dataTime", int(analysistime.strftime("%H%M")))
        ecc.codes_set(h, "forecastTime", forecastTime)
        ecc.codes_set(h, "centre", 86)
        ecc.codes_set(h, "generatingProcessIdentifier", args.producer_id)
        ecc.codes_set(h, "discipline", 0)
        ecc.codes_set(h, "parameterCategory", pcat)
        ecc.codes_set(h, "parameterNumber", pnum)
        ecc.codes_set(h, "productDefinitionTemplateNumber", pdtn)
        ##Some if else for tmin and tmax parameter
        
        ecc.codes_set(h, "typeOfFirstFixedSurface", 103)
        ecc.codes_set(h, "level", levelvalue) 
        ecc.codes_set(h, "packingType", "grid_ccsds")
        ecc.codes_set(h, "indicatorOfUnitOfTimeRange", iouot)  # hours
        ecc.codes_set(h, "typeOfGeneratingProcess", 2)  # deterministic forecast
        ecc.codes_set(h, "typeOfProcessedData", 2)  # analysis and forecast products
        ecc.codes_set_values(h, tdata.flatten())
        ecc.codes_write(h, fp)
        forecastTime_prev = forecastTime
    ecc.codes_release(h)
             

def write_grib(args, analysistime, forecasttime, data):
    if args.output.startswith("s3://"):
        openfile = fsspec.open(
            "simplecache::{}".format(args.output),
            "wb",
            s3={
                "anon": False,
                "key": os.environ["S3_ACCESS_KEY_ID"],
                "secret": os.environ["S3_SECRET_ACCESS_KEY"],
                "client_kwargs": {"endpoint_url": "https://eris.fmi.fi"},
            },
        )
        with openfile as fpout:
            write_grib_message(fpout, args, analysistime, forecasttime, data)
    else:
        with open(args.output, "wb") as fpout:
            write_grib_message(fpout, args, analysistime, forecasttime, data)
            
    print(f"Wrote file {args.output}")



