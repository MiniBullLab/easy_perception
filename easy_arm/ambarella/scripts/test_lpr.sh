#!/bin/sh
export PATH="$PATH:/bin:/sbin"

init.sh --yuv480p
modprobe imx316_mipi bus_id=0x2 vinc_id=0x1 trigger_mode=1
modprobe ar0234_mipi bus_id=0x0 vinc_id=0x0

test_tuning -a &

rmmod ambarella_fb
modprobe ambarella_fb resolution=240x180 mode=clut8bpp

modprobe cavalry
cavalry_load -f /lib/firmware/cavalry.bin -r

test_encode --resource-cfg /usr/local/bin/scripts/cv25_vin0_wuxga_vin1_240x180_tof_linear.lua --hdmi 720p --raw-capture 1 --enc-raw-yuv 1 --enc-mode 0 --extra-raw-dram-buf-num 16 --vsync-detect-disable 1

test_encode -C -h 1920x1200 -b 2 -e
rtsp_server &
test_mempart -m 4 -f
test_mempart -m 4 -s 0x04000000

# init.sh --na;modprobe imx274_mipi;test_tuning -a &
# rmmod ambarella_fb; modprobe ambarella_fb resolution=1920x1080 mode=clut8bpp
# modprobe cavalry;cavalry_load -f /lib/firmware/cavalry.bin -r
# test_encode --resource-cfg /data/lpr/cv22_vin0_1080p_linear_ssd_lpr.lua --hdmi 1080p --ors 1920x1080 --enc-dummy-latency 5
#test_encode --resource-cfg /data/lpr/cv22_vin0_1080p_linear_ssd_lpr.lua --hdmi 1080p --ors 1920x1080
#test_encode -A -H 1080p -e
#rtsp_server &
#test_mempart -m 4 -f
#test_mempart -m 4 -s 0x04009000
export LD_LIBRARY_PATH=/data:$LD_LIBRARY_PATH

if [ ! -d "./result_video" ]; then
    mkdir ./result_video
fi
./test_lpr
