#!/bin/sh

export LD_LIBRARY_PATH=/data:$LD_LIBRARY_PATH

if [ ! -d "/data/glog_file" ]; then
    mkdir /data/glog_file
fi

/data/send_lpr > /data/glog_file/send_lpr.log 2>&1 &
