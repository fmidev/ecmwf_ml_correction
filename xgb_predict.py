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
    parser.add_argument("--output_file_t2", action="store", type=str, required=True)
    parser.add_argument("--output_file_td2", action="store", type=str, required=True)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--disable_multiprocessing", action="store_true", default=False)
    
    args = parser.parse_args()       
    return args

                                                                                                    
def main():
    args = parse_command_line()

    #Read NWP data and create fetures array
    st = time.time()
    all_features, features_list = create_features_data(args)
    print("Reading NWP data for ML features takes:", round(time.time()-st, 1), "seconds")

    #ML prediction for temperature and dewpoint
    mlt = time.time()
    ml_predictions_t2 = ml_predict(args, all_features, features_list, "temperature")
    ml_predictions_td2 = ml_predict(args, all_features, features_list, "dewpoint")
    print("Producing ML forecasts for temperature and dewpoint takes:", round(time.time()-mlt, 1), "seconds")

    #Gridding
    oit = time.time()
    grid, lons, lats, background_t2, leadtimes, analysistime, forecasttime, lc, topo = read_grid(args, "temperature")
    background0 = copy.copy(background_t2)
    background0[background0 != 0] = 0
    points = get_points(grid, lc, args)
    diff_t2 = interpolate(grid, points, background0[0], ml_predictions_t2, args, lc)
    output_t2, forecasttime_out = ml_corrected_forecasts(forecasttime, background_t2, diff_t2, "temperature")
    grid, lons, lats, background_td2, leadtimes, analysistime, forecasttime, lc, topo = read_grid(args, "dewpoint")
    diff_td2 = interpolate(grid, points, background0[0], ml_predictions_td2, args, lc)
    output_td2, _ = ml_corrected_forecasts(forecasttime, background_td2, diff_td2, "dewpoint")
    #Set that output of dewpoint cant be higher than output of temperature
    for i in range(0, len(output_td2)):
        td_gt_t2 = output_td2[i] > output_t2[i]
        output_td2[i][td_gt_t2] = output_t2[i][td_gt_t2] - 0.0001
    print("Interpolating forecasts takes:", round(time.time()-oit, 1), "seconds")

    #Write corrected forecasts to grib files
    write_grib(args, analysistime, forecasttime_out, output_t2, args.output_file_t2, "temperature")
    write_grib(args, analysistime, forecasttime_out, output_td2, args.output_file_td2, "dewpoint")

if __name__ == "__main__":
    main()

