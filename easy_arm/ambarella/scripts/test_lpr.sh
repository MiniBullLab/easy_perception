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

if [ ! -d "./result_video" ]; then
    mkdir ./result_video
fi
./test_lpr
