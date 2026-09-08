FROM rockylinux/rockylinux:8

ARG BRANCH=main

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm && \
    dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python311 python311-pip python39-setuptools eccodes git && \
    dnf -y clean all && rm -rf /var/cache/dnf && \
    git clone -b "$BRANCH" --single-branch https://github.com/fmidev/ecmwf_ml_correction.git

WORKDIR /ecmwf_ml_correction

ARG TA_TAG=20260812
ARG TD_TAG=20260812
ARG TMAX_TAG=20260812
ARG TMIN_TAG=20260812
ARG station_TAG=2026

ARG S3_HOST=https://lake.fmi.fi

ADD $S3_HOST/ml-models/ecmwf-ml-correction/ec_mos_lsm.grib /meps_ml_correction
ADD $S3_HOST/ml-models/ecmwf-ml-correction/ec_mos_topography.grib /meps_ml_correction
ADD $S3_HOST/ml-models/ecmwf-ml-correction/stations_nearest_$station_TAG.csv /ecmwf_ml_correction/all_stations.csv
ADD $S3_HOST/ml-models/ecmwf-ml-correction/XGB_T2_$TA_TAG.json /meps_ml_correction/XGB_T2.json
ADD $S3_HOST/ml-models/ecmwf-ml-correction/XGB_TD_$TD_TAG.json /meps_ml_correction/XGB_TD.json
ADD $S3_HOST/ml-models/ecmwf-ml-correction/XGB_TMAX_$TMAX_TAG.json /meps_ml_correction/XGB_TMAX.json
ADD $S3_HOST/ml-models/ecmwf-ml-correction/XGB_TMIN_$TMIN_TAG.json /meps_ml_correction/XGB_TMIN.json

RUN chmod 644 ec_mos_lsm.grib && \
    chmod 644 ec_mos_topography.grib && \
    chmod 644 all_stations.csv && \
    chmod 644 XGB_T2.json && \
    chmod 644 XGB_TD.json && \
    chmod 644 XGB_TMAX.json && \
    chmod 644 XGB_TMIN.json && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    python3 -m pip --no-cache-dir install -r requirements.txt
