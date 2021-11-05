#!/bin/sh

ntpdate 10.0.0.102 &

sleep 5

export LD_LIBRARY_PATH=/data:$LD_LIBRARY_PATH

if [ ! -d "./glogfile" ]; then
    mkdir ./glogfile
fi
./test_lpr
