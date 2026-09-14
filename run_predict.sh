#!/bin/bash
# Run predict script for the ecmwf ml correction forecasts: tempeature and dewpoint 
#E.g. ./run_predict.sh 2026090700 "path/to/temperature_file.grib2" "path/to/dewpoint_file.grib2" 122

PYTHON=python3
ANALYSIS_TIME=$1 #YYYYMMDDHH
OUTPUT_FILE_T2=$2
OUTPUT_FILE_TD=$3

PRODUCER_ID=$4 #122 for preop, 120 for oper

echo "ANALYSIS_TIME:" $ANALYSIS_TIME
echo "OUTPUT_FILE_T2:" $OUTPUT_FILE_T2
echo "OUTPUT_FILE_TD:" $OUTPUT_FILE_TD

# Local (static) data (fetched from S3 when run in openshift)
TOPO="ec_mos_topography.grib"
LC="ec_mos_lsm.grib"
MODEL_TA="XGB_T2.json"
MODEL_TD="XGB_TD.json"
MODEL_TMAX="XGB_TMAX.json"
MODEL_TMIN="XGB_TMIN.json"
STATIONS_FILE="all_stations.csv"
   
# Data from S3 eris
if [ $PRODUCER_ID == 120 ]; then
    bucket="s3://routines-data/ecmwf-ml-correction/prod/"$ANALYSIS_TIME"00/"
else
    bucket="s3://routines-data/ecmwf-ml-correction/preop/"$ANALYSIS_TIME"00/"
fi

P0=$bucket"P-PA_0.grib"
T2=$bucket"T-K_0.grib" #ECMWF:llä ground or surface level nolla on lämpötilalle 2m:ssä ja tuulelle 10 m:ssä
TD2=$bucket"TD-K_0.grib"
U10=$bucket"U-MS_0.grib"
V10=$bucket"V-MS_0.grib"
LCC=$bucket"NL-0TO1_0.grib"
MCC=$bucket"NM-0TO1_0.grib"
SKT=$bucket"SKT-K_0.grib"
TMAX=$bucket"TMAX-K_0.grib"
TMIN=$bucket"TMIN-K_0.grib"
T925=$bucket"T-K_925.grib"
T_ENSMEAN=$bucket"T-MEAN-K_0.grib"

#Generating ml corrected forecasts for both temperature and dewpoint at the same time
$PYTHON xgb_predict.py --topography_data $TOPO --landseacover_data $LC --p_data $P0 --t2_data $T2 --td2_data $TD2 --u10_data $U10 --v10_data $V10 --lcc_data $LCC --mcc_data $MCC --skt_data $SKT --tmax_data $TMAX --tmin_data $TMIN --t925_data $T925 --t_ensmean_data $T_ENSMEAN --model_ta $MODEL_TA --model_td $MODEL_TD --model_tmax $MODEL_TMAX --model_tmin $MODEL_TMIN --stations_list $STATIONS_FILE --analysis_time $ANALYSIS_TIME --producer_id $PRODUCER_ID --output_file_t2 $OUTPUT_FILE_T2 --output_file_td2 $OUTPUT_FILE_TD
