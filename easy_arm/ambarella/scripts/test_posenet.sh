#!/bin/sh
export PATH="$PATH:/bin:/sbin"
export LD_LIBRARY_PATH=/data:$LD_LIBRARY_PATH

./posenet_test
