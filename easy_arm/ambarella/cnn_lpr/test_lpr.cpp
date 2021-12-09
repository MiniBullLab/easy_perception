
#include <signal.h>
#include <stdint.h>
#include <sys/prctl.h>

#include "utility/utils.h"

#include "cnn_lpr/common/common_process.h"
#include "cnn_lpr/lpr/det_process.h"
#include "cnn_lpr/lpr/lpr_process.h"

#include "cnn_lpr/drivers/image_acquisition.h"
#include "cnn_lpr/drivers/tof_316_acquisition.h"

#include "cnn_lpr/tof/tof_data_process.h"
#include "cnn_lpr/image/vibebgs.h"

#include "cnn_lpr/network/tcp_process.h"
#include "cnn_lpr/network/network_process.h"
#include "cnn_lpr/common/save_data_process.h"

#include "cnn_runtime/det2d/denet.h"

#define DEFAULT_SSD_LAYER_ID		(0)
#define DEFAULT_LPR_LAYER_ID		(0)

#define CLASS_NUMBER (1)

#define FRAME_HEADER1	(0x55)
#define FRAME_HEADER2	(0xAA)
#define YUV_DATA_LENGTH	(8 + 4 + 4 + 4 + 4 + IMAGE_YUV_SIZE)
#define TOTAL_YUV_SIZE	(2 + 1 + 4 + YUV_DATA_LENGTH)
#define TOF_DATA_LENGTH (8 + 4 + 4 + 4 + 4 + TOF_SIZE)
#define TOTAL_TOF_SIZE  (2 + 1 + 4 + TOF_DATA_LENGTH)

#define TIME_MEASURE_LOOPS			(20)

#define IS_LPR_RUN
//#define IS_DENET_RUN
#define IS_PC_RUN

const static std::string model_path = "./denet.bin";
const static std::vector<std::string> input_name = {"data"};
const static std::vector<std::string> output_name = {"det_output0", "det_output1", "det_output2"};
const char* class_name[CLASS_NUMBER] = {"car"};

static std::vector<int> list_has_lpr;
static std::vector<bbox_param_t> list_lpr_bbox;
static float lpr_confidence = 0;
static std::string lpr_result = "";

static pthread_mutex_t result_mutex;

volatile int has_lpr = 0;
volatile int run_flag = 1;
volatile int run_lpr = 0;
volatile int run_denet = 0;

static TCPProcess tcp_process;
static TOF316Acquisition tof_geter;
static ImageAcquisition image_geter;
static SaveDataProcess save_process;
static NetWorkProcess network_process;

#if defined(ONLY_SEND_DATA)
static void *send_tof_pthread(void *thread_params)
{
	uint64_t debug_time = 0;
	unsigned char send_data[TOTAL_TOF_SIZE];
	int length = 0;
	long time_stamp = 0;
	int data_type = 0, width = 0, height = 0;
	prctl(PR_SET_NAME, "send_tof_pthread");
	tof_geter.set_up();
	while(run_flag)
	{
		TIME_MEASURE_START(1);
		tof_geter.get_tof_Z(&send_data[31]);
		send_data[0] = FRAME_HEADER1;
		send_data[1] = FRAME_HEADER2;
		send_data[2] = 0x07;
		length = TOF_DATA_LENGTH;
		fill_data(&send_data[3], length);

		time_stamp = get_time_stamp();
		fill_data(&send_data[7], (time_stamp >> 32) & 0xFFFFFFFF);
		fill_data(&send_data[11], time_stamp & 0xFFFFFFFF);

		data_type = 1;
		fill_data(&send_data[15], data_type);

		width = DEPTH_WIDTH;
		fill_data(&send_data[19], width);

		height = DEPTH_HEIGTH;
		fill_data(&send_data[23], height);

		tcp_process.send_data(send_data, TOTAL_TOF_SIZE);

		TIME_MEASURE_END("[send_tof_pthread] cost time", 1);
	}
	// pthread_detach(pthread_self());
	LOG(WARNING) << "send_tof_pthread quit";
	return NULL;
}

static void *send_yuv_pthread(void *thread_params)
{
	uint64_t debug_time = 0;
	unsigned char send_data[TOTAL_YUV_SIZE];
	int length = 0;
	long time_stamp = 0;
	int data_type = 0, width = 0, height = 0;
	prctl(PR_SET_NAME, "send_yuv_pthread");
	while(run_flag)
	{
		TIME_MEASURE_START(1);
 		image_geter.get_yuv(&send_data[31]);
		send_data[0] = FRAME_HEADER1;
		send_data[1] = FRAME_HEADER2;
		send_data[2] = 0x06;
		length = YUV_DATA_LENGTH;
		fill_data(&send_data[3], length);

		time_stamp = get_time_stamp();
		fill_data(&send_data[7], (time_stamp >> 32) & 0xFFFFFFFF);
		fill_data(&send_data[11], time_stamp & 0xFFFFFFFF);

		data_type = 0;
		fill_data(&send_data[15], data_type);

		width = IMAGE_WIDTH;
		fill_data(&send_data[19], width);

		height = IMAGE_HEIGHT;
		fill_data(&send_data[23], height);

		tcp_process.send_data(send_data, TOTAL_YUV_SIZE);

		TIME_MEASURE_END("[send_yuv_pthread] cost time", 1);
	}
	// pthread_detach(pthread_self());
	LOG(WARNING) << "send_yuv_pthread quit";
	return NULL;
}

#endif

static int send_count = 0;

static void merge_all_result(const int in_out_result)
{
	int final_result = in_out_result;
	size_t lpr_count = list_lpr_bbox.size();
	int lpr_in_out = 0;
	int sum_count = 0;
	float lpr_sum = 0;
	std::vector<bbox_param_t> result_bbox;
	if(lpr_count >= 3)
	{
		for (size_t i = list_has_lpr.size() - 9; i >= 0; i--)
		{
			lpr_sum += list_has_lpr[i];
			sum_count++;
			if(sum_count > 30)
			{
				break;
			}
		}
		lpr_sum = lpr_sum / sum_count;
		if(lpr_sum > 0.5f)
		{
			lpr_in_out = 1;
		}
		else if(lpr_sum <= 0.5f && lpr_sum > 0.1)
		{
			lpr_in_out = 2;
		}
		LOG(WARNING) << "lpr_in_out: " << lpr_in_out << " " << lpr_sum;
	}

	result_bbox = bbox_list_process(list_lpr_bbox);
	LOG(WARNING) << "result bbox: " <<  result_bbox.size();

	if(send_count == 0 && final_result == 2)
	{
		final_result = 0;
	}
	if(send_count == 0 && lpr_in_out == 2)
	{
		lpr_in_out = 0;
	}

	if(send_count == 1 && final_result == 1)
	{
		final_result = 0;
	}
	if(send_count == 1 && lpr_in_out == 1)
	{
		lpr_in_out = 0;
	}

	if(final_result > 0)
	{
		if(lpr_result != "" && lpr_confidence > 0)
		{
			network_process.send_result(lpr_result, final_result);
			send_count = 1;
			lpr_confidence = 0;
		}
		else if(final_result == 1 && lpr_result != "")
		{
			network_process.send_result(lpr_result, final_result);
			lpr_result = "";
			lpr_confidence = 0;
		}
		if(final_result == 2)
		{
			send_count = 0;
		}
	}
	else if(lpr_in_out > 0)
	{
		if(lpr_result != "" && lpr_confidence > 0)
		{
			network_process.send_result(lpr_result, lpr_in_out);
			send_count = 1;
			lpr_confidence = 0;
		}
		else if(final_result == 1 && lpr_result != "")
		{
			network_process.send_result(lpr_result, lpr_in_out);
			lpr_result = "";
			lpr_confidence = 0;
		}
	}
	list_has_lpr.clear();
	list_lpr_bbox.clear();
}

static void *run_lpr_pthread(void *lpr_param_thread)
{
	int rval;
	ssd_lpr_thread_params_t *lpr_param =
		(ssd_lpr_thread_params_t*)lpr_param_thread;
	global_control_param_t *G_param = lpr_param->G_param;
	LPR_ctx_t LPR_ctx;

	// Detection result param
	bbox_param_t bbox_param[MAX_DETECTED_LICENSE_NUM];
	draw_plate_list_t draw_plate_list;
	uint16_t license_num = 0;
	license_list_t license_result;
	state_buffer_t *ssd_mid_buf;
	ea_img_resource_data_t * data = NULL;
	ea_tensor_t *img_tensor = NULL;

	// Time mesurement
	uint64_t start_time = 0;
	uint64_t debug_time = 0;
	float sum_time = 0.0f;
	float average_license_num = 0.0f;
	uint32_t loop_count = 1;
	uint32_t debug_en = G_param->debug_en;

	prctl(PR_SET_NAME, "lpr_pthread");

	do {
		memset(&LPR_ctx, 0, sizeof(LPR_ctx));
		memset(&draw_plate_list, 0, sizeof(draw_plate_list));
		memset(&bbox_param, 0, sizeof(bbox_param));

		LPR_ctx.img_h = lpr_param->height;
		LPR_ctx.img_w = lpr_param->width;
		RVAL_OK(init_LPR(&LPR_ctx, G_param));
		RVAL_OK(alloc_single_state_buffer(&G_param->ssd_result_buf, &ssd_mid_buf));

		while (run_flag) {
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
			while(run_lpr > 0)
#endif
			{
				RVAL_OK(lpr_critical_resource(&license_num, bbox_param, ssd_mid_buf, G_param));
				start_time = gettimeus();
				data = (ea_img_resource_data_t *)ssd_mid_buf->img_resource_addr;
				if (license_num == 0) {
#if defined(OFFLINE_DATA)
					for (int i = 0; i < data->tensor_num; i++) {
						if (data->tensor_group[i]) {
							ea_tensor_free(data->tensor_group[i]);
							data->tensor_group[i] = NULL;
						}
					}
					free(data->tensor_group);
					data->tensor_group = NULL;
					data->led_group = NULL;
#else
					RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, data));
#endif
					continue;
				}
				img_tensor = data->tensor_group[DEFAULT_LPR_LAYER_ID];
				if (G_param->abort_if_preempted) {
					pthread_mutex_lock(&G_param->vp_access_lock);
				}
				RVAL_OK(LPR_run_vp_preprocess(&LPR_ctx, img_tensor, license_num, (void*)bbox_param));
				if (G_param->abort_if_preempted) {
					pthread_mutex_unlock(&G_param->vp_access_lock); // unlock to let SSD run during LPR ARM time
				}

				RVAL_OK(LPR_run_arm_preprocess(&LPR_ctx, license_num));
				if (G_param->abort_if_preempted) {
					pthread_mutex_lock(&G_param->vp_access_lock);
				}
				RVAL_OK(LPR_run_vp_recognition(&LPR_ctx, license_num, &license_result));
#ifdef IS_SHOW
				draw_overlay_preprocess(&draw_plate_list, &license_result, bbox_param, G_param);
				TIME_MEASURE_START(debug_en);
				RVAL_OK(set_overlay_image(img_tensor, &draw_plate_list));
				TIME_MEASURE_END("[LPR] LPR draw overlay time", debug_en);
#endif
				if (G_param->abort_if_preempted) {
					pthread_mutex_unlock(&G_param->vp_access_lock);
				}

				if(license_result.license_num > 0)
				{
					pthread_mutex_lock(&result_mutex);
					bbox_param_t lpr_bbox = {0};
					lpr_bbox.norm_min_x = bbox_param[0].norm_min_x * LPR_ctx.img_w;
					lpr_bbox.norm_min_y = bbox_param[0].norm_min_y * LPR_ctx.img_h;
					lpr_bbox.norm_max_x = bbox_param[0].norm_max_x * LPR_ctx.img_w;
					lpr_bbox.norm_max_y = bbox_param[0].norm_max_y * LPR_ctx.img_h;

#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
					list_lpr_bbox.push_back(lpr_bbox);
#endif

					LOG(WARNING) << "bbox: " << lpr_bbox.norm_min_x << " " \
					<< lpr_bbox.norm_min_y << " " \
				    << lpr_bbox.norm_max_x << " " \
					<< lpr_bbox.norm_max_y;
					
					LOG(INFO) << "LPR:"  << license_result.license_info[0].text << " " << license_result.license_info[0].conf;
					if (license_result.license_info[0].conf > DEFAULT_LPR_CONF_THRES && \
						strlen(license_result.license_info[0].text) == CHINESE_LICENSE_STR_LEN && \
						license_result.license_info[0].conf > lpr_confidence)
						{
							lpr_result = license_result.license_info[0].text;
							lpr_confidence = license_result.license_info[0].conf;
							LOG(WARNING) << "LPR:"  << lpr_result << " " << lpr_confidence;
						}
					pthread_mutex_unlock(&result_mutex);
				}
				else
				{

				}

#if defined(OFFLINE_DATA)
				for (int i = 0; i < data->tensor_num; i++) {
					if (data->tensor_group[i]) {
						ea_tensor_free(data->tensor_group[i]);
						data->tensor_group[i] = NULL;
					}
				}
				free(data->tensor_group);
				data->tensor_group = NULL;
				data->led_group = NULL;
#else
				RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, data));
#endif
				sum_time += (gettimeus() - start_time);
				++loop_count;
				average_license_num += license_num;
				if (loop_count == TIME_MEASURE_LOOPS) {
					float average_time1 = sum_time / (1000 * TIME_MEASURE_LOOPS);
					float average_time2 = (average_license_num > 0.0f) ? (sum_time / (1000 * average_license_num)) : 0.0f;
					LOG(INFO) << "[" << TIME_MEASURE_LOOPS  << "loops] LPR average time license_num " << " " << average_license_num / TIME_MEASURE_LOOPS;
					LOG(WARNING) << "average time:"<< average_time1 << " per license cost time:" << average_time2;
					sum_time = 0;
					loop_count = 1;
					average_license_num = license_num;
				}
			}
			usleep(20000);
		}
	} while (0);
	do {
		run_flag = 0;
		free_single_state_buffer(ssd_mid_buf);
		LPR_deinit(&LPR_ctx);
		LOG(WARNING) << "LPR thread quit.";
	} while (0);

	return NULL;
}

static void wait_vp_available(pthread_mutex_t *vp_access_lock)
{
	pthread_mutex_lock(vp_access_lock);
	pthread_mutex_unlock(vp_access_lock);
	return;
}

static void *run_ssd_pthread(void *ssd_thread_params)
{
	int rval = 0;
	unsigned long long int frame_number = 0;
	uint32_t i = 0;
	ssd_lpr_thread_params_t *ssd_param =
		(ssd_lpr_thread_params_t*)ssd_thread_params;
	global_control_param_t *G_param = ssd_param->G_param;
	// SSD param
	SSD_ctx_t SSD_ctx;
	int ssd_result_num = 0;
	bbox_param_t scaled_license_plate;
	state_buffer_t *ssd_mid_buf;
	ssd_net_final_result_t ssd_net_result;
	bbox_list_t bbox_list;

	// image related
	ea_tensor_t *img_tensor = NULL;
	ea_img_resource_data_t data;

	// Time measurement
	uint64_t start_time = 0;
	uint64_t debug_time = 0;
	float sum_time = 0.0f;
	uint32_t loop_count = 1;
	uint32_t debug_en = G_param->debug_en;

    uint32_t dsp_pts = 0;

	// cv::Mat bgr(ssd_param->height * 2 / 3, ssd_param->width, CV_8UC3);

#if defined(OFFLINE_DATA)
	ea_tensor_t **tensors = NULL;
	int tensor_num;
	size_t img_shape[4] = {1, 1, ssd_param->height * 3 / 2, ssd_param->width};
	// void *yuv_data = ea_tensor_data_for_read(dst, EA_CPU);
#endif


	prctl(PR_SET_NAME, "ssd_pthread");

	do {
		memset(&ssd_net_result, 0, sizeof(ssd_net_result));
		memset(&data, 0, sizeof(data));
		memset(&SSD_ctx, 0, sizeof(SSD_ctx_t));
		memset(&scaled_license_plate, 0, sizeof(scaled_license_plate));

		RVAL_OK(init_ssd(&SSD_ctx, G_param, ssd_param->height, ssd_param->width));

		ssd_net_result.dproc_ssd_result = (dproc_ssd_detection_output_result_t *)
			malloc(SSD_ctx.vp_result_info.max_dproc_ssd_result_num *
			sizeof(dproc_ssd_detection_output_result_t));
		RVAL_ASSERT(ssd_net_result.dproc_ssd_result != NULL);
		RVAL_OK(alloc_single_state_buffer(&G_param->ssd_result_buf, &ssd_mid_buf));
 
		while (run_flag) {
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
			while(run_lpr > 0)
#endif
			{
				start_time = gettimeus();

				TIME_MEASURE_START(1);
#if defined(OFFLINE_DATA)
				cv::Mat src_image;
				save_process.get_image_yuv(src_image);
				if(src_image.empty())
				{
					LOG(ERROR) << "SSD get image fail!";
					continue;
				}
				img_tensor = ea_tensor_new(EA_U8, img_shape, ssd_param->pitch);
				mat2tensor_yuv_nv12(src_image, img_tensor);
				tensors = (ea_tensor_t **)malloc(sizeof(ea_tensor_t *) * 1);
				RVAL_ASSERT(tensors != NULL);
				memset(tensors, 0, sizeof(ea_tensor_t *) * 1);
				// led_status = (int *)malloc(sizeof(int) * 1);
				// RVAL_ASSERT(led_status != NULL);
				// memset(led_status, 0, sizeof(int) * 1);
				// led_status[0] = 0;
				tensor_num = 1;
				data.mono_pts = 0;
				data.dsp_pts = 0;
				data.tensor_group = tensors;
				data.tensor_num = tensor_num;
				data.led_group = NULL;
				data.tensor_group[DEFAULT_SSD_LAYER_ID] = img_tensor;

				// std::stringstream filename_image;
                // filename_image << "/data/save_data/" << "image_" << frame_number << ".jpg";
				// RVAL_OK(ea_tensor_to_jpeg(img_tensor, EA_TENSOR_COLOR_MODE_YUV_NV12, filename_image.str().c_str()));
#else
				RVAL_OK(ea_img_resource_hold_data(G_param->img_resource, &data));
				RVAL_ASSERT(data.tensor_group != NULL);
				RVAL_ASSERT(data.tensor_num >= 1);
				img_tensor = data.tensor_group[DEFAULT_SSD_LAYER_ID];
				dsp_pts = data.dsp_pts;
#endif
				TIME_MEASURE_END("[SSD]run_lpr get yuv cost time", 1);
				// std::cout << "ssd size:" << ea_tensor_shape(img_tensor)[0] << " " << ea_tensor_shape(img_tensor)[1] << " " << ea_tensor_shape(img_tensor)[2] << " " << ea_tensor_shape(img_tensor)[3] << std::endl;
				// SAVE_TENSOR_IN_DEBUG_MODE("SSD_pyd.jpg", img_tensor, debug_en);
				// if(frame_number % 80 == 0)
				// {
				// 	has_lpr = 0;
				// }
				frame_number++;

#ifdef IS_SAVE
				TIME_MEASURE_START(debug_en);
				// RVAL_OK(tensor2mat_yuv2bgr_nv12(img_tensor, bgr));
				// save_process.put_image_data(bgr);
				std::stringstream filename_image;
                filename_image << save_process.get_image_save_dir() << "image_" << frame_number << ".jpg";
				RVAL_OK(ea_tensor_to_jpeg(img_tensor, EA_TENSOR_COLOR_MODE_YUV_NV12, filename_image.str().c_str()));
				TIME_MEASURE_END("[SSD] yuv to bgr time", debug_en);
#endif

				TIME_MEASURE_START(debug_en);
				// std::cout << "ssd size:" << ea_tensor_shape(SSD_ctx.net_input.tensor)[2] << " " << ea_tensor_shape(SSD_ctx.net_input.tensor)[3] << " " << ea_tensor_pitch(SSD_ctx.net_input.tensor) << std::endl;
				RVAL_OK(ea_cvt_color_resize(img_tensor, SSD_ctx.net_input.tensor,
					EA_COLOR_YUV2BGR_NV12, EA_VP));
				TIME_MEASURE_END("[SSD] preprocess time", debug_en);

				TIME_MEASURE_START(debug_en);
				if (G_param->abort_if_preempted) {
					wait_vp_available(&G_param->vp_access_lock);
				}
				rval = ssd_net_run_vp_forward(&SSD_ctx.ssd_net_ctx);
				if (rval < 0) {
					if (rval == -EAGAIN) {
						do {
							if (G_param->abort_if_preempted) {
								wait_vp_available(&G_param->vp_access_lock);
							}
							rval = ea_net_resume(SSD_ctx.ssd_net_ctx.net);
							if (rval == -EINTR) {
								printf("SSD network interrupt by signal in resume\n");
								break;
							}
						} while (rval == -EAGAIN);
					} else if (rval == -EINTR) {
						printf("SSD network interrupt by signal in run\n");
						break;
					} else {
						printf("ssd_net_run_vp_forward failed, ret: %d\n", rval);
						rval = -1;
						break;
					}
				}
			
				ea_tensor_sync_cache(SSD_ctx.ssd_net_ctx.output_loc_tensor, EA_VP, EA_CPU);
				ea_tensor_sync_cache(SSD_ctx.ssd_net_ctx.output_conf_tensor, EA_VP, EA_CPU);
				TIME_MEASURE_END("[SSD] network time", debug_en);

				TIME_MEASURE_START(debug_en);
				ssd_net_result.ssd_det_num = 0;
				memset(&ssd_net_result.labels[0][0], 0,
					SSD_NET_MAX_LABEL_NUM * SSD_NET_MAX_LABEL_LEN);
				memset(ssd_net_result.dproc_ssd_result, 0,
					SSD_ctx.vp_result_info.max_dproc_ssd_result_num *
					sizeof(dproc_ssd_detection_output_result_t));
				RVAL_OK(ssd_net_run_arm_nms(&SSD_ctx.ssd_net_ctx,
					SSD_ctx.vp_result_info.loc_dram_addr,
					SSD_ctx.vp_result_info.conf_dram_addr, &ssd_net_result));
				TIME_MEASURE_END("[SSD] ARM NMS time", debug_en);

				TIME_MEASURE_START(debug_en);
				ssd_result_num = min(ssd_net_result.ssd_det_num, MAX_DETECTED_LICENSE_NUM);
				bbox_list.bbox_num = min(ssd_result_num, MAX_OVERLAY_PLATE_NUM);
				ssd_critical_resource(ssd_net_result.dproc_ssd_result, &data,
					bbox_list.bbox_num, ssd_mid_buf, G_param);

				if(bbox_list.bbox_num > 0)
				{
					has_lpr = 1;
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
					pthread_mutex_lock(&result_mutex);
					list_has_lpr.push_back(1);
					pthread_mutex_unlock(&result_mutex);
#endif
				}
				else
				{
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
					pthread_mutex_lock(&result_mutex);
					list_has_lpr.push_back(0);
					pthread_mutex_unlock(&result_mutex);
#endif 
				}
				LOG(WARNING) << "lpr box count:" << bbox_list.bbox_num;

				TIME_MEASURE_END("[SSD] post-process time", debug_en);

#ifdef IS_SHOW
				for (i = 0; i < bbox_list.bbox_num; ++i) {
					upscale_normalized_rectangle(ssd_net_result.dproc_ssd_result[i].bbox.x_min,
					ssd_net_result.dproc_ssd_result[i].bbox.y_min,
					ssd_net_result.dproc_ssd_result[i].bbox.x_max,
					ssd_net_result.dproc_ssd_result[i].bbox.y_max,
					DRAW_LICNESE_UPSCALE_W, DRAW_LICNESE_UPSCALE_H, &scaled_license_plate);
					bbox_list.bbox[i].norm_min_x = scaled_license_plate.norm_min_x;
					bbox_list.bbox[i].norm_min_y = scaled_license_plate.norm_min_y;
					bbox_list.bbox[i].norm_max_x = scaled_license_plate.norm_max_x;
					bbox_list.bbox[i].norm_max_y = scaled_license_plate.norm_max_y;

					LOG(INFO) << "lpr :" << ssd_net_result.dproc_ssd_result[i].bbox.x_min << " " << ssd_net_result.dproc_ssd_result[i].bbox.y_min << " " << bbox_list.bbox[i].norm_max_x << " " << bbox_list.bbox[i].norm_max_y;
				}

				RVAL_OK(set_overlay_bbox(&bbox_list));
				RVAL_OK(show_overlay(dsp_pts));
#endif

				sum_time += (gettimeus() - start_time);
				++loop_count;
				if (loop_count == TIME_MEASURE_LOOPS) {
					LOG(WARNING) << "SSD average time [per " << TIME_MEASURE_LOOPS << " loops]:" << sum_time / (1000 * TIME_MEASURE_LOOPS) << "ms";
					sum_time = 0;
					loop_count = 1;
				}
			}
#if defined(OFFLINE_DATA) && defined(IS_PC_RUN) && defined(IS_LPR_RUN)
            TIME_MEASURE_START(1);
			cv::Mat src_image;
			save_process.get_image(src_image);
			if(src_image.empty())
			{
				LOG(ERROR) << "SSD get image fail!";
				continue;
			}
			has_lpr = 0;
			TIME_MEASURE_END("[SSD] get yuv cost time", 1);
#else
            has_lpr = 0;
			usleep(20000);
#endif
		}
	} while (0);
	do {
		run_flag = 0;
		if (ssd_net_result.dproc_ssd_result != NULL) {
			free(ssd_net_result.dproc_ssd_result);
		}
		ssd_net_deinit(&SSD_ctx.ssd_net_ctx);
		free_single_state_buffer(ssd_mid_buf);
		LOG(WARNING) << "SSD thread quit.";
	} while (0);

	return NULL;
}

static void *run_denet_pthread(void *thread_params)
{
	int rval = 0;
	uint64_t debug_time = 0;
	std::vector<std::vector<float>> boxes;
    DeNet denet_process;
	bbox_list_t bbox_list;
	prctl(PR_SET_NAME, "run_denet_pthread");
	if(denet_process.init(model_path, input_name, output_name, CLASS_NUMBER) < 0)
    {
		LOG(ERROR) << "DeNet init fail!";
        return NULL;
    }
	while(run_flag)
	{
		cv::Mat src_image;
		TIME_MEASURE_START(1);
#if defined(OFFLINE_DATA)
		save_process.get_image(src_image);
		if(src_image.empty())
		{
			LOG(ERROR) << "DeNet get image fail!";
			continue;
		}
#else
		image_geter.get_image(src_image);
		if(src_image.empty())
		{
			LOG(ERROR) << "DeNet get image fail!";
			continue;
		}
#endif
		boxes = denet_process.run(src_image);
		TIME_MEASURE_END("[run_denet_pthread] cost time", 1);
		bbox_list.bbox_num = min(boxes.size(), MAX_OVERLAY_PLATE_NUM);
		for (size_t i = 0; i < bbox_list.bbox_num; ++i)
	    {
            float xmin = boxes[i][0];
            float ymin = boxes[i][1];
            float xmax = xmin + boxes[i][2];
            float ymax = ymin + boxes[i][3];
            int type = boxes[i][4];
            float confidence = boxes[i][5];
			bbox_list.bbox[i].norm_min_x = xmin / src_image.cols;
			bbox_list.bbox[i].norm_min_y = ymin / src_image.rows;
			bbox_list.bbox[i].norm_max_x = xmax / src_image.cols;
			bbox_list.bbox[i].norm_max_y = ymax / src_image.rows;
			LOG(WARNING) << "car :" << xmin << " " << ymin << " " << xmax << " " << ymax << " " << confidence;
	    }
		LOG(WARNING) << "car count:" <<bbox_list.bbox_num;
#ifdef IS_SHOW
		RVAL_OK(set_car_bbox(&bbox_list));
#endif
	}
	run_denet = 0;
	LOG(WARNING) << "run_denet_pthread quit！";
	return NULL;
}

static void *process_recv_pthread(void *thread_params)
{
	uint64_t debug_time = 0;
	prctl(PR_SET_NAME, "process_recv_pthread");
	while(run_flag)
	{
		int result = network_process.process_recv();
		if(result == 200)
		{
			run_denet = 0;
			run_lpr = 0;
			run_flag = 0;
#if defined(OFFLINE_DATA)
			save_process.offline_stop();
#endif
		}
	}
	LOG(WARNING) << "process_recv_pthread quit！";
	return NULL;
}

static void process_pc_pthread(const global_control_param_t *G_param)
{
	uint64_t start_time = 0;
	float sum_time = 0.0f;
	float average_license_num = 0.0f;
	uint32_t loop_count = 1;
	
	uint64_t debug_time = 0;
	uint32_t debug_en = G_param->debug_en;
	bool first_save = true;
	int bg_point_count = 0;
	// int is_in = -1;
	unsigned long long int process_number = 0;
	unsigned long long int no_process_number = 0;
	cv::Mat filter_map;
	cv::Mat pre_map;
	cv::Mat bg_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH),CV_8UC1);
	cv::Mat depth_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH),CV_8UC1);
	std::vector<int> point_cout_list;

	cv::Mat img_bgmodel;
	cv::Mat img_output;
	IBGS *bgs = new ViBeBGS();

#if !defined(IS_LPR_RUN)
	lpr_confidence = 1;
	lpr_result = "12345678";
	has_lpr = 1;
#endif

	point_cout_list.clear();

#if defined(OFFLINE_DATA)
	save_process.get_tof_depth_map(depth_map);
#else
	tof_geter.get_tof_depth_map(depth_map);
#endif
	// cv::medianBlur(depth_map, bg_map, 3);
	cv::GaussianBlur(depth_map, bg_map, cv::Size(9, 9), 3.5, 3.5);
	cv::imwrite("./bg.png", bg_map);

	while(run_flag > 0)
	{
		start_time = gettimeus();
		TIME_MEASURE_START(1);
#if defined(OFFLINE_DATA)
		save_process.get_tof_depth_map(depth_map);
#else
		tof_geter.get_tof_depth_map(depth_map);
#endif
		TIME_MEASURE_END("[point_cloud] get TOF cost time", 1);

		TIME_MEASURE_START(debug_en);
		cv::GaussianBlur(depth_map, filter_map, cv::Size(9, 9), 3.5, 3.5);
		TIME_MEASURE_END("[point_cloud] filtering cost time", debug_en);

		TIME_MEASURE_START(debug_en);
		bgs->process(filter_map, img_output, img_bgmodel);
		bg_point_count = static_cast<int>(cv::sum(img_output / 255)[0]);
		LOG(WARNING) << "bg_point_count:" << bg_point_count;
		TIME_MEASURE_END("[point_cloud] bgs cost time", debug_en);

		// if(process_number % 1 == 0)
		// {
		// 	std::stringstream filename;
		// 	filename << "point_cloud" << process_number << ".png";
		// 	cv::imwrite(filename.str(), img_output);
		// 	// dump_bin(filename.str(), src_cloud);
		// }
		// process_number++;

		if(bg_point_count > 50)
		{
			no_process_number = 0;
			run_lpr = 1;
			tof_geter.set_up();
#ifdef IS_SAVE
			if(first_save)
			{
				if(save_process.start() >= 0)
				{
					first_save = false;
					save_process.put_tof_data(pre_map);
				}	
			}
			if(!first_save)
				save_process.put_tof_data(depth_map);
#endif
			// if(has_lpr == 1)
			{
				point_cout_list.push_back(bg_point_count);
			}
		}
		else
		{
			no_process_number++;
			if(no_process_number % 10 == 0)
			{
				int in_out_result = vote_in_out(point_cout_list);
				int point_count = compute_depth_map(bg_map, filter_map);
				LOG(WARNING) << "final point_count:" << point_count << " " << in_out_result;
				if(in_out_result == 1 && point_count >= 400)
				{
					in_out_result = 1;
				}
				else if(in_out_result == 1 && point_count < 100)
				{
					in_out_result = 2;
				}
				else if(in_out_result == 2 && point_count >= 400)
				{
					in_out_result = 1;
				}
				LOG(WARNING) << "in_out_result:" << in_out_result;
				pthread_mutex_lock(&result_mutex);
				merge_all_result(in_out_result);
				pthread_mutex_unlock(&result_mutex);
				point_cout_list.clear();
				process_number = 0;
				no_process_number = 0;
				has_lpr = 0;
				run_lpr = 0;
				tof_geter.set_sleep();
#if !defined(IS_LPR_RUN)
				lpr_confidence = 1;
	            lpr_result = "12345678";
				has_lpr = 1;
#endif
#ifdef IS_SAVE
				if(!first_save)
				{
					save_process.stop();
					first_save = true;
				}
#endif
				LOG(WARNING) << "no process";
			}
		}
		TIME_MEASURE_END("[point_cloud] cost time", debug_en);

		if(process_number % 10 == 0)
		{
			pre_map = depth_map.clone();
		}
		process_number++;
		// usleep(25000);

		sum_time += (gettimeus() - start_time);
		++loop_count;
		if (loop_count == TIME_MEASURE_LOOPS) {
			LOG(WARNING) << "PC average time [per " << TIME_MEASURE_LOOPS << " loops]:" << sum_time / (1000 * TIME_MEASURE_LOOPS) << "ms";
			sum_time = 0;
			loop_count = 1;
		}
	}
	run_denet = 0;
	run_lpr = 0;
	run_flag = 0;
	delete bgs;
    bgs = NULL;
	LOG(WARNING) << "stop point cloud process";
}

static int start_all(global_control_param_t *G_param)
{
	int rval = 0;
	pthread_t process_recv_pthread_id = 0;
	pthread_t denet_pthread_id = 0;
	pthread_t ssd_pthread_id = 0;
	pthread_t lpr_pthread_id = 0;
	ssd_lpr_thread_params_t lpr_thread_params;
	ssd_lpr_thread_params_t ssd_thread_params;

	ea_tensor_t *img_tensor = NULL;
	ea_img_resource_data_t data;

	list_has_lpr.clear();
	list_lpr_bbox.clear();

	if(network_process.start() < 0)
	{
		rval = -1;
		run_flag = 0;
		LOG(ERROR) << "start network fail!";
		return rval;
	}
	else
	{
		LOG(INFO) << "start network success";
	}

#if defined(OFFLINE_DATA)
	save_process.offline_start();
#else
	if(tof_geter.start() < 0)
	{
		rval = -1;
		run_flag = 0;
		LOG(ERROR) << "start tof fail!";
		return rval;
	}
	else
	{
		LOG(INFO) << "start tof success";
	}

#if defined(IS_DENET_RUN)
	if(image_geter.start() < 0)
	{
		rval = -1;
		run_flag = 0;
		LOG(ERROR) << "start image fail!";
		return rval;
	}
	else
	{
		LOG(INFO) << "start image success";
	}
#endif
#endif

	do {
		pthread_mutex_init(&result_mutex, NULL);
		memset(&lpr_thread_params, 0 , sizeof(lpr_thread_params));
		memset(&data, 0, sizeof(data));
		RVAL_OK(ea_img_resource_hold_data(G_param->img_resource, &data));
		RVAL_ASSERT(data.tensor_group != NULL);
		RVAL_ASSERT(data.tensor_num >= 1);
		img_tensor = data.tensor_group[DEFAULT_LPR_LAYER_ID];
		lpr_thread_params.height = ea_tensor_shape(img_tensor)[2];
		lpr_thread_params.width = ea_tensor_shape(img_tensor)[3];
		lpr_thread_params.pitch = ea_tensor_pitch(img_tensor);
		std::cout << "lpr:" << lpr_thread_params.height << " " << lpr_thread_params.width << " " << lpr_thread_params.pitch << std::endl;
		lpr_thread_params.G_param = G_param;
		img_tensor = data.tensor_group[DEFAULT_SSD_LAYER_ID];
		ssd_thread_params.height = ea_tensor_shape(img_tensor)[2];
		ssd_thread_params.width = ea_tensor_shape(img_tensor)[3];
		ssd_thread_params.pitch = ea_tensor_pitch(img_tensor);
		ssd_thread_params.G_param = G_param;
		std::cout << "ssd:" << ssd_thread_params.height << " " << ssd_thread_params.width << " " << ssd_thread_params.pitch << std::endl;
		RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &data));
#if defined(IS_LPR_RUN)
		rval = pthread_create(&ssd_pthread_id, NULL, run_ssd_pthread, (void*)&ssd_thread_params);
		RVAL_ASSERT(rval == 0);
		rval = pthread_create(&lpr_pthread_id, NULL, run_lpr_pthread, (void*)&lpr_thread_params);
		RVAL_ASSERT(rval == 0);
#endif

#if defined(IS_DENET_RUN)
		rval = pthread_create(&denet_pthread_id, NULL, run_denet_pthread, NULL);
		RVAL_ASSERT(rval == 0);
#endif
		rval = pthread_create(&process_recv_pthread_id, NULL, process_recv_pthread, NULL);
		RVAL_ASSERT(rval == 0);
	} while (0);
	LOG(INFO) << "start_ssd_lpr success";

#if defined(IS_PC_RUN)
	process_pc_pthread(G_param);
#endif

	if (lpr_pthread_id > 0) {
		pthread_join(lpr_pthread_id, NULL);
		lpr_pthread_id = 0;
	}
	LOG(WARNING) << "lpr pthread release";
	if (ssd_pthread_id > 0) {
		pthread_join(ssd_pthread_id, NULL);
		ssd_pthread_id = 0;
	}
	LOG(WARNING) << "ssd pthread release";
	if (denet_pthread_id > 0) {
		pthread_join(denet_pthread_id, NULL);
		denet_pthread_id = 0;
	}
	LOG(WARNING) << "denet pthread release";
#if defined(OFFLINE_DATA)
	save_process.offline_stop();
#else
	tof_geter.stop();
#if defined(IS_DENET_RUN)
	image_geter.stop();
#endif
	save_process.stop();
#endif
	network_process.stop();
	if (process_recv_pthread_id > 0) {
		pthread_join(process_recv_pthread_id, NULL);
		process_recv_pthread_id = 0;
	}
	LOG(WARNING) << "process_recv_pthread pthread release";
	pthread_mutex_destroy(&result_mutex);
	LOG(WARNING) << "Main thread quit";
	return rval;
}

#if defined(ONLY_SAVE_DATA)
static void *run_tof_pthread(void *thread_params)
{
	uint64_t debug_time = 0;
	//unsigned char tof_data[TOF_SIZE];
	cv::Mat depth_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH),CV_8UC1);
	prctl(PR_SET_NAME, "run_tof_pthread");
	tof_geter.set_up();
	while(run_flag)
	{
		TIME_MEASURE_START(1);
		// tof_geter.get_tof_Z(tof_data);
		// save_process.save_tof_z(depth_map);
		tof_geter.get_tof_depth_map(depth_map);
		save_process.save_depth_map(depth_map);
		TIME_MEASURE_END("[run_tof_pthread] cost time", 1);
	}
	// pthread_detach(pthread_self());
	LOG(WARNING) << "run_tof_pthread quit";
	return NULL;
}

static void *run_image_pthread(void *thread_params)
{
	uint64_t debug_time = 0;
	prctl(PR_SET_NAME, "run_image_pthread");
	while(run_flag)
	{
		cv::Mat src_image;
		TIME_MEASURE_START(1);
		image_geter.get_image(src_image);
		save_process.save_image(src_image);
		TIME_MEASURE_END("[run_image_pthread] cost time", 1);
	}
	LOG(WARNING) << "run_image_pthread quit";
	return NULL;
}

#endif

#if defined(ONLY_SAVE_DATA) || defined(ONLY_SEND_DATA)
static int start_all()
{
	int rval = 0;
	if(tof_geter.start() < 0)
	{
		rval = -1;
		run_flag = 0;
		LOG(ERROR) << "start tof fail!";
	}
	LOG(INFO) << "start tof success";
	if(image_geter.start() < 0)
	{
		rval = -1;
		run_flag = 0;
		LOG(ERROR) << "start image fail!";
	}
	LOG(INFO) << "start image success";
	return rval;
}
#endif

static void sigstop(int signal_number)
{
	run_denet = 0;
	run_lpr = 0;
	run_flag = 0;
#if defined(OFFLINE_DATA)
	save_process.offline_stop();
#endif
	LOG(WARNING) << "sigstop msg, exit";
}

static void SignalHandle(const char* data, int size) {
    std::string str = data;
	run_denet = 0;
	run_lpr = 0;
	run_flag = 0;
#if defined(OFFLINE_DATA)
	save_process.offline_stop();
#endif
    LOG(FATAL) << str;
}

int main(int argc, char **argv)
{
	int rval = 0;
	global_control_param_t G_param;

	google::InitGoogleLogging(argv[0]);

	FLAGS_log_dir = "/data/glog_file";
	// google::SetLogDestination(google::GLOG_ERROR, "/data/glogfile/logerror");
	google::InstallFailureSignalHandler();
	google::InstallFailureWriter(&SignalHandle); 

	FLAGS_stderrthreshold = 1;
	FLAGS_colorlogtostderr = true; 
	FLAGS_logbufsecs = 5;    //缓存的最大时长，超时会写入文件
	FLAGS_max_log_size = 10; //单个日志文件最大，单位M
	FLAGS_logtostderr = false; //设置为true，就不会写日志文件了
	// FLAGS_alsologtostderr = true;
	FLAGS_minloglevel = 0;
	FLAGS_stop_logging_if_full_disk = true;

	signal(SIGINT, sigstop);
	signal(SIGQUIT, sigstop);
	signal(SIGTERM, sigstop);
#if defined(ONLY_SAVE_DATA)
    pthread_t tof_pthread_id = 0;
	pthread_t image_pthread_id = 0;
	save_process.init_data();
	save_process.init_save_dir();
	if(tof_geter.open_tof() == 0 && image_geter.open_camera() == 0)
	{
		if(start_all() >= 0)
		{
			run_flag = 1;
			rval = pthread_create(&tof_pthread_id, NULL, run_tof_pthread, NULL);
			if(rval >= 0)
			{
				rval = pthread_create(&image_pthread_id, NULL, run_image_pthread, NULL);
				if(rval < 0)
				{
					run_flag = 0;
					LOG(ERROR) << "create pthread fail!";
				}
			}
			else
			{
				run_flag = 0;
				LOG(ERROR) << "create pthread fail!";
			}
			if (tof_pthread_id > 0) {
				pthread_join(tof_pthread_id, NULL);
			}
			if (image_pthread_id > 0) {
				pthread_join(image_pthread_id, NULL);
			}
			tof_geter.stop();
			image_geter.stop();
			LOG(WARNING) << "Main thread quit";
		}
		else
		{
			LOG(ERROR) << "start_all fail!";
		}
	}
#elif defined(ONLY_SEND_DATA)
	pthread_t tof_pthread_id = 0;
	pthread_t image_pthread_id = 0;
	if(tof_geter.open_tof() == 0 && image_geter.open_camera() == 0)
	{
		if(tcp_process.socket_init() >= 0 && start_all() >= 0)
		{
			if(tcp_process.accept_connect() >= 0)
			{
				run_flag = 1;
				rval = pthread_create(&tof_pthread_id, NULL, send_tof_pthread, NULL);
				if(rval >= 0)
				{
					rval = pthread_create(&image_pthread_id, NULL, send_yuv_pthread, NULL);
					if(rval < 0)
					{
						run_flag = 0;
						LOG(ERROR) << "create pthread fail!";
					}
				}
				else
				{
					run_flag = 0;
					LOG(ERROR) << "create pthread fail!";
				}
				if (tof_pthread_id > 0) {
					pthread_join(tof_pthread_id, NULL);
				}
				if (image_pthread_id > 0) {
					pthread_join(image_pthread_id, NULL);
				}
			}
			tof_geter.stop();
			image_geter.stop();
			LOG(WARNING) << "Main thread quit";
		}
		else
		{
			LOG(ERROR) << "start_all fail!";
		}
	}
#elif defined(OFFLINE_DATA)
	if(network_process.init_network() < 0)
	{
		rval = -1;
		run_flag = 0;
		return -1;
	}
	LOG(INFO) << "net init success";
	save_process.set_save_dir("/data/save_data/2021_12_08_09_26_56_1/image/", "/data/save_data/2021_12_08_09_26_56_1/tof/");
	do {
		RVAL_OK(init_param(&G_param));
		RVAL_OK(env_init(&G_param));
		RVAL_OK(start_all(&G_param));
	}while(0);
	env_deinit(&G_param);
#else
	if(tof_geter.open_tof() == 0 /*&& image_geter.open_camera() == 0*/)
	{
		if(network_process.init_network() < 0)
		{
			rval = -1;
			run_flag = 0;
			return -1;
		}
		LOG(INFO) << "net init success";
		save_process.init_data();
		do {
			RVAL_OK(init_param(&G_param));
			RVAL_OK(env_init(&G_param));
			RVAL_OK(start_all(&G_param));
		}while(0);
		env_deinit(&G_param);
	}
#endif
	LOG(INFO) << "All Quit";
	google::ShutdownGoogleLogging();
	return rval;
}


