init.sh --na;modprobe imx274_mipi;test_tuning -a &
rmmod ambarella_fb; modprobe ambarella_fb resolution=1920x1080 mode=clut8bpp
modprobe cavalry;cavalry_load -f /lib/firmware/cavalry.bin -r
test_encode --resource-cfg /data/car_license/cv22_vin0_1080p_linear_ssd_lpr.lua --hdmi 1080p --ors 1920x1080 --enc-dummy-latency 5
#test_encode --resource-cfg /data/car_license/cv22_vin0_1080p_linear_ssd_lpr.lua --hdmi 1080p --ors 1920x1080
#test_encode -A -H 1080p -e
#rtsp_server &
#test_mempart -m 4 -f
#test_mempart -m 4 -s 0x04009000
test_ssd_lpr -b /data/car_license/mobilenetv1_ssd_cavalry.bin --in data --out mbox_loc --out mbox_conf_flatten --class 2 --background_id 0 --top_k 50 --nms_top_k 100 --nms_threshold 0.45 --pri_threshold 0.3 --priorbox /data/car_license/lpr_priorbox_fp32.bin -i 0 -t 1 -b /data/car_license/segfree_inception_cavalry.bin --in data --out prob -b /data/car_license/LPHM_cavalry.bin --in data --out dense
