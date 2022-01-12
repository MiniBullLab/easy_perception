#!/bin/sh

export LD_LIBRARY_PATH=/data:$LD_LIBRARY_PATH

if [ ! -d "/data/glog_file" ]; then
    mkdir /data/glog_file
fi

if [ ! -d "/data/save_data" ]; then
    mkdir /data/save_data
fi

/data/record_lpr > /data/glog_file/record_lpr.log 2>&1 &
