init.sh --na;modprobe imx274_mipi;test_tuning -a &
rmmod ambarella_fb; modprobe ambarella_fb resolution=1920x1080 mode=clut8bpp
modprobe cavalry;cavalry_load -f /lib/firmware/cavalry.bin -r
test_encode --resource-cfg /data/lpr/cv22_vin0_1080p_linear_ssd_lpr.lua --hdmi 1080p --ors 1920x1080 --enc-dummy-latency 5
#test_encode --resource-cfg /data/lpr/cv22_vin0_1080p_linear_ssd_lpr.lua --hdmi 1080p --ors 1920x1080
#test_encode -A -H 1080p -e
#rtsp_server &
#test_mempart -m 4 -f
#test_mempart -m 4 -s 0x04009000
export LD_LIBRARY_PATH=/data:$LD_LIBRARY_PATH
./test_lpr
