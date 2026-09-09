#Predict script that follows meps_ml_correction predict script
import sys
import copy
import time
import argparse
from helper_functions import create_features_data
from helper_functions import ml_predict
from helper_functions import read_grid
from helper_functions import get_points
from helper_functions import interpolate
from helper_functions import write_grib
from helper_functions import ml_corrected_forecasts

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", action="store", type=str, required=True)
    parser.add_argument("--topography_data", action="store", type=str, required=True)
    parser.add_argument("--landseacover_data", action="store", type=str, required=True)
    parser.add_argument("--p_data", action="store", type=str, required=True)
    parser.add_argument("--t2_data", action="store", type=str, required=True)
    parser.add_argument("--td2_data", action="store", type=str, required=True)
    parser.add_argument("--u10_data", action="store", type=str, required=True)
    parser.add_argument("--v10_data", action="store", type=str, required=True)
    parser.add_argument("--lcc_data", action="store", type=str, required=True)
    parser.add_argument("--mcc_data", action="store", type=str, required=True)
    parser.add_argument("--skt_data", action="store", type=str, required=True)
    parser.add_argument("--tmax_data", action="store", type=str, required=True)
    parser.add_argument("--tmin_data", action="store", type=str, required=True)
    parser.add_argument("--t925_data", action="store", type=str, required=True)
    parser.add_argument("--t_ensmean_data", action="store", type=str, required=True)
    parser.add_argument("--model_ta", action="store", type=str, required=True)
    parser.add_argument("--model_td", action="store", type=str, required=True)
    parser.add_argument("--model_tmax", action="store", type=str, required=True)
    parser.add_argument("--model_tmin", action="store", type=str, required=True)
    parser.add_argument("--stations_list", action="store", type=str, required=True)
    parser.add_argument("--analysis_time", action="store", type=str, required=True)
    parser.add_argument("--producer_id", action="store", type=int, required=True)
    parser.add_argument("--output", action="store", type=str, required=True)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--disable_multiprocessing", action="store_true", default=False)
    
    args = parser.parse_args()
    
    allowed_params = ["dewpoint","temperature"]
    if args.parameter not in allowed_params:
        print("Error: parameter must be one of: {}".format(allowed_params))
        sys.exit(1)
        
    return args

                                                                                                    
def main():
    args = parse_command_line()

    #Read NWP data and create fetures array
    st = time.time()
    all_features, features_list = create_features_data(args)
    print("Reading NWP data for", args.parameter, "takes:", round(time.time()-st, 1), "seconds")

    #ML prediction
    mlt = time.time()
    ml_predictions = ml_predict(args, all_features, features_list, args.parameter)
    print("Producing ML forecasts takes:", round(time.time()-mlt, 1), "seconds")

    #Gridding
    oit = time.time()
    grid, lons, lats, background, leadtimes, analysistime, forecasttime, lc, topo = read_grid(args, args.parameter)
    background0 = copy.copy(background)
    background0[background0 != 0] = 0
    points = get_points(grid, lc, args)
    diff = interpolate(grid, points, background0[0], ml_predictions, args, lc)
    output, forecasttime = ml_corrected_forecasts(forecasttime, background, diff, args.parameter)
    print("Interpolating forecasts takes:", round(time.time()-oit, 1), "seconds")

    #Write corrected forecasts to grib file
    write_grib(args, analysistime, forecasttime, output)

if __name__ == "__main__":
    main()

