
#include <signal.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/prctl.h>

#include "utility/utils.h"

#include "cnn_lpr/common/common_process.h"
#include "cnn_lpr/det2d/denetv2.h"
#include "cnn_lpr/lpr/det_lpr_process.h"
#include "cnn_lpr/lpr/lpr_process.h"
#include "cnn_lpr/lpr/Rec.h"

#include "cnn_lpr/drivers/image_acquisition.h"
#include "cnn_lpr/drivers/tof_316_acquisition.h"

#include "cnn_lpr/tof/tof_data_process.h"
#include "cnn_lpr/image/vibebgs.h"

#include "cnn_lpr/network/tcp_process.h"
#include "cnn_lpr/network/network_process.h"
#include "cnn_lpr/common/save_data_process.h"

#define CLASS_NUMBER (1)

#define FRAME_HEADER1	(0x55)
#define FRAME_HEADER2	(0xAA)
#define YUV_DATA_LENGTH	(8 + 4 + 4 + 4 + 4 + IMAGE_YUV_SIZE)
#define TOTAL_YUV_SIZE	(2 + 1 + 4 + YUV_DATA_LENGTH)
#define TOF_DATA_LENGTH (8 + 4 + 4 + 4 + 4 + TOF_SIZE)
#define TOTAL_TOF_SIZE  (2 + 1 + 4 + TOF_DATA_LENGTH)

#define TIME_MEASURE_LOOPS			(20)

#define IS_LPR_RUN
//#define IS_CAR_RUN
#define IS_PC_RUN

#if defined(OLD_CODE)
#define DEFAULT_SSD_LAYER_ID		(1)
#define DEFAULT_LPR_LAYER_ID		(0)
#else
#define DEFAULT_SSD_LAYER_ID		(0)
#define DEFAULT_LPR_LAYER_ID		(0)
#endif

static has_car_list_t list_has_lpr;
static has_lpr_list_t list_has_car;
static bbox_list_t list_lpr_bbox;
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

	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;

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

	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;

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

static int in_send_count = 0;
static int out_send_count = 0;

static void merge_all_result(const int in_out_result)
{
	int final_result = in_out_result;
	int lpr_in_out = 0;
	int has_car = 0;
	float car_sum = 0;
	float lpr_sum = 0;
	bbox_list_t result_bbox;
	result_bbox.bbox_num = 0;
#if defined(IS_CAR_RUN)
	if(list_has_car.bbox_num >= 3)
	{
		car_sum = 0;
		for (int i = 0; i < list_has_car.bbox_num; i++)
		{
			car_sum += list_has_car.has[i];
		}
		car_sum = car_sum / list_has_car.bbox_num;
		if(car_sum > 0.5f)
		{
			has_car = 1;
		}
		else
		{
			has_car = 0;
		}
		LOG(WARNING) << "has_car: " << has_car << " " << car_sum;
	}
#endif
	if(list_lpr_bbox.bbox_num >= 3)
	{
		LOG(WARNING) << "input bbox: " <<  list_lpr_bbox.bbox_num;
		bbox_list_process(&list_lpr_bbox, &result_bbox);
		LOG(WARNING) << "result bbox: " << result_bbox.bbox_num;
		// for (int i = 0; i < list_lpr_bbox.bbox_num; ++i) {
		// 	list_lpr_bbox.bbox[i].norm_min_x = list_lpr_bbox.bbox[i].norm_min_x / IMAGE_WIDTH;
		// 	list_lpr_bbox.bbox[i].norm_min_y = list_lpr_bbox.bbox[i].norm_min_y / IMAGE_HEIGHT;
		// 	list_lpr_bbox.bbox[i].norm_max_x = list_lpr_bbox.bbox[i].norm_max_x / IMAGE_WIDTH;
		// 	list_lpr_bbox.bbox[i].norm_max_y = list_lpr_bbox.bbox[i].norm_max_y / IMAGE_HEIGHT;
		// }
		// for (int i = 0; i < result_bbox.bbox_num; ++i) {
		// 	result_bbox.bbox[i].norm_min_x = result_bbox.bbox[i].norm_min_x / IMAGE_WIDTH;
		// 	result_bbox.bbox[i].norm_min_y = result_bbox.bbox[i].norm_min_y / IMAGE_HEIGHT;
		// 	result_bbox.bbox[i].norm_max_x = result_bbox.bbox[i].norm_max_x / IMAGE_WIDTH;
		// 	result_bbox.bbox[i].norm_max_y = result_bbox.bbox[i].norm_max_y / IMAGE_HEIGHT;
		// }
		// set_overlay_bbox(&list_lpr_bbox);
		// set_car_bbox(&result_bbox);
		// show_overlay(0);
	}
	if(list_has_lpr.bbox_num > 20)
	{
		lpr_sum = 0;
		for (int i = 0; i < list_has_lpr.bbox_num; i++)
		{
			lpr_sum += list_has_lpr.has[i];
		}
		lpr_sum = lpr_sum / list_has_lpr.bbox_num;
		if(lpr_sum > 0.5f)
		{
			lpr_in_out = 1;
		}
		else if(lpr_sum > 0)
		{
			lpr_in_out = 2;
		}
		LOG(WARNING) << "lpr_in_out: " << lpr_in_out << " " << lpr_sum;
	}

	if(final_result > 0)
	{
		if(in_send_count == 0 && final_result == 2)
		{
			return;
		}
		if(in_send_count == 1 && final_result == 1 && result_bbox.bbox_num == 1)
		{
			return;
		}
		if(lpr_result != "" && lpr_confidence > 0)
		{
			LOG(WARNING) << "lpr_confidence:" << lpr_confidence; 
			network_process.send_result(lpr_result, final_result);
			lpr_result = "";
			lpr_confidence = 0;
			if(final_result == 1)
			{
				in_send_count = 1;
			}
			else
			{
				out_send_count = 1;
			}
		}
		else if(lpr_in_out > 0 && final_result == 1)
		{
			network_process.send_result("001", 3);
			in_send_count = 1;
		}
		else if(lpr_in_out > 0 && final_result == 2)
		{
			network_process.send_result("001", 4);
			out_send_count = 1;
		}

		if(final_result == 2 || out_send_count == 1)
		{
			in_send_count = 0;
			out_send_count = 0;
		}
	}
	else if(lpr_in_out > 0)
	{
		if(in_send_count == 0 && lpr_in_out == 2)
		{
			return;
		}
		if(in_send_count == 1 && lpr_in_out == 1 && result_bbox.bbox_num == 1)
		{
			return;
		}
		if(lpr_result != "" && lpr_confidence > 0)
		{
			LOG(WARNING) << "lpr_confidence:" << lpr_confidence; 
			network_process.send_result(lpr_result, lpr_in_out);
			lpr_result = "";
			lpr_confidence = 0;
			if(lpr_in_out == 1)
			{
				in_send_count = 1;
			}
			else
			{
				out_send_count = 1;
			}
		}
		else if(lpr_in_out == 1)
		{
			network_process.send_result("001", 3);
			in_send_count = 1;
		}
		else if(lpr_in_out == 2)
		{
			network_process.send_result("001", 4);
			out_send_count = 1;
		}

		if(out_send_count == 1)
		{
			in_send_count = 0;
			out_send_count = 0;
		}
	}

	list_has_lpr.bbox_num = 0;
	list_lpr_bbox.bbox_num = 0;
	list_has_car.bbox_num = 0;
}

#if defined(OLD_CODE)
static void *run_lpr_pthread(void *param_thread)
{
	int rval;
	lpr_thread_params_t *lpr_param =
		(lpr_thread_params_t*)param_thread;
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

	list_lpr_bbox.bbox_num = 0;

	std::cout << "lpr size:" << lpr_param->width << " " << lpr_param->height << " " << lpr_param->pitch << std::endl;

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
				if(lpr_critical_resource(&license_num, bbox_param, ssd_mid_buf, G_param) < 0)
					continue;
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
				draw_overlay_preprocess(&draw_plate_list, &license_result, bbox_param, debug_en);
				TIME_MEASURE_START(debug_en);
				RVAL_OK(set_overlay_image(img_tensor, &draw_plate_list));
				TIME_MEASURE_END("[LPR] LPR draw overlay time", debug_en);
#endif
				if (G_param->abort_if_preempted) {
					pthread_mutex_unlock(&G_param->vp_access_lock);
				}

				if(license_result.license_num > 0)
				{
					size_t char_len = strlen(license_result.license_info[0].text);
					LOG(INFO) << "LPR:"  << license_result.license_info[0].text << " " \
								<< license_result.license_info[0].conf << " "
								<< char_len;
					if (license_result.license_info[0].conf > DEFAULT_LPR_CONF_THRES && \
						(char_len == 9 || char_len == 10))
						{
							pthread_mutex_lock(&result_mutex);
							if(license_result.license_info[0].conf > lpr_confidence)
							{
								lpr_result = license_result.license_info[0].text;
								lpr_confidence = license_result.license_info[0].conf;
								LOG(WARNING) << "LPR:"  << lpr_result << " " << lpr_confidence;
							}
							bbox_param_t lpr_bbox = {0};
							lpr_bbox.norm_min_x = bbox_param[0].norm_min_x * lpr_param->width;
							lpr_bbox.norm_min_y = bbox_param[0].norm_min_y * lpr_param->height;
							lpr_bbox.norm_max_x = bbox_param[0].norm_max_x * lpr_param->width;
							lpr_bbox.norm_max_y = bbox_param[0].norm_max_y * lpr_param->height;
							if(list_lpr_bbox.bbox_num < MAX_OVERLAY_PLATE_NUM)
							{
								list_lpr_bbox.bbox[list_lpr_bbox.bbox_num++] = lpr_bbox;
							}
							else
							{
								LOG(INFO) << "not add lpr bbox";
							}
							LOG(INFO) << "LPR bbox: " << lpr_bbox.norm_min_x << " " \
														<< lpr_bbox.norm_min_y << " " \
														<< lpr_bbox.norm_max_x << " " \
														<< lpr_bbox.norm_max_y;
							pthread_mutex_unlock(&result_mutex);
						}
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
					LOG(WARNING) << "LPR average time:"<< average_time1 << " per license cost time:" << average_time2;
					sum_time = 0;
					loop_count = 1;
					average_license_num = license_num;
				}
			}
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
			lpr_result = "";
			lpr_confidence = 0;
			list_lpr_bbox.bbox_num = 0;
			usleep(20000);
#endif
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

static void *run_det_lpr_pthread(void *thread_params)
{
	int rval = 0;
	unsigned long long int frame_number = 0;
	lpr_thread_params_t *det_lpr_param =
		(lpr_thread_params_t*)thread_params;
	global_control_param_t *G_param = det_lpr_param->G_param;
	// SSD param
	SSD_ctx_t SSD_ctx;
	ssd_net_final_result_t ssd_net_result;
	int ssd_result_num = 0;
	bbox_param_t scaled_license_plate;

	state_buffer_t *ssd_mid_buf;
	bbox_list_t bbox_list = {0};

	// image related
	ea_tensor_t *img_tensor = NULL;
	ea_img_resource_data_t data;
	uint32_t dsp_pts = 0;

	// Time measurement
	uint64_t start_time = 0;
	uint64_t debug_time = 0;
	float sum_time = 0.0f;
	uint32_t loop_count = 1;
	uint32_t debug_en = G_param->debug_en;

	uint32_t list_index = 0;
	list_has_lpr.bbox_num = 0;

#if defined(OFFLINE_DATA)
	ea_tensor_t **tensors = NULL;
	int tensor_num;
	size_t img_shape[4] = {1, 1, det_lpr_param->height * 3 / 2, det_lpr_param->width};
	// void *yuv_data = ea_tensor_data_for_read(dst, EA_CPU);
#endif

	std::cout << "det_lpr size:" << det_lpr_param->width << " " << det_lpr_param->height << " " << det_lpr_param->pitch << std::endl;

	prctl(PR_SET_NAME, "det_lpr_pthread");

	do {
		memset(&data, 0, sizeof(data));
		memset(&bbox_list, 0, sizeof(bbox_list_t));

		memset(&SSD_ctx, 0, sizeof(SSD_ctx_t));
		memset(&ssd_net_result, 0, sizeof(ssd_net_result));
		memset(&scaled_license_plate, 0, sizeof(scaled_license_plate));
		RVAL_OK(init_ssd(&SSD_ctx, G_param, det_lpr_param->height, det_lpr_param->width));
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
					LOG(ERROR) << "det_lpr get image fail!";
					continue;
				}
				img_tensor = ea_tensor_new(EA_U8, img_shape, det_lpr_param->pitch);
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
				TIME_MEASURE_END("[det_lpr]run_lpr get yuv cost time", 1);
				// std::cout << "det_lpr size:" << ea_tensor_shape(img_tensor)[0] << " " << ea_tensor_shape(img_tensor)[1] << " " << ea_tensor_shape(img_tensor)[2] << " " << ea_tensor_shape(img_tensor)[3] << std::endl;
				// SAVE_TENSOR_IN_DEBUG_MODE("det_lpr_pyd.jpg", img_tensor, debug_en);
				// if(frame_number % 80 == 0)
				// {
				// 	has_lpr = 0;
				// }
				// frame_number++;

				TIME_MEASURE_START(debug_en);
				// std::cout << "det_lpr size:" << ea_tensor_shape(SSD_ctx.net_input.tensor)[2] << " " << ea_tensor_shape(SSD_ctx.net_input.tensor)[3] << " " << ea_tensor_pitch(SSD_ctx.net_input.tensor) << std::endl;
				RVAL_OK(ea_cvt_color_resize(img_tensor, SSD_ctx.net_input.tensor, EA_COLOR_YUV2BGR_NV12, EA_VP));
				TIME_MEASURE_END("[det_lpr] preprocess time", debug_en);

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
								printf("det_lpr network interrupt by signal in resume\n");
								break;
							}
						} while (rval == -EAGAIN);
					} else if (rval == -EINTR) {
						printf("det_lpr network interrupt by signal in run\n");
						break;
					} else {
						printf("ssd_net_run_vp_forward failed, ret: %d\n", rval);
						rval = -1;
						break;
					}
				}
			
				ea_tensor_sync_cache(SSD_ctx.ssd_net_ctx.output_loc_tensor, EA_VP, EA_CPU);
				ea_tensor_sync_cache(SSD_ctx.ssd_net_ctx.output_conf_tensor, EA_VP, EA_CPU);
				TIME_MEASURE_END("[det_lpr] network time", debug_en);

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

				TIME_MEASURE_START(debug_en);
				ssd_result_num = min(ssd_net_result.ssd_det_num, MAX_DETECTED_LICENSE_NUM);
				bbox_list.bbox_num = min(ssd_result_num, MAX_OVERLAY_PLATE_NUM);
				ssd_critical_resource(ssd_net_result.dproc_ssd_result, &data,
					bbox_list.bbox_num, ssd_mid_buf, G_param);

				pthread_mutex_lock(&result_mutex);
				if(bbox_list.bbox_num > 0)
				{
					has_lpr = 1;
				}
				else
				{
					has_lpr = 0;
				}
				if(list_has_lpr.bbox_num < MAX_OVERLAY_PLATE_NUM)
					list_has_lpr.bbox_num++;
				list_has_lpr.has[list_index++] = has_lpr;
				list_index = list_index % MAX_OVERLAY_PLATE_NUM;
				pthread_mutex_unlock(&result_mutex);
				LOG(WARNING) << "lpr box count:" << bbox_list.bbox_num;

				TIME_MEASURE_END("[det_lpr] post-process time", debug_en);

#ifdef IS_SHOW
				for (int i = 0; i < bbox_list.bbox_num; ++i) {
					upscale_normalized_rectangle(ssd_net_result.dproc_ssd_result[i].bbox.x_min,
					ssd_net_result.dproc_ssd_result[i].bbox.y_min,
					ssd_net_result.dproc_ssd_result[i].bbox.x_max,
					ssd_net_result.dproc_ssd_result[i].bbox.y_max,
					DRAW_LICNESE_UPSCALE_W, DRAW_LICNESE_UPSCALE_H, &scaled_license_plate);
					bbox_list.bbox[i].norm_min_x = scaled_license_plate.norm_min_x;
					bbox_list.bbox[i].norm_min_y = scaled_license_plate.norm_min_y;
					bbox_list.bbox[i].norm_max_x = scaled_license_plate.norm_max_x;
					bbox_list.bbox[i].norm_max_y = scaled_license_plate.norm_max_y;
					bbox_list.bbox[i].score = ssd_net_result.dproc_ssd_result[i].score;
					LOG(INFO) << "lpr :" << ssd_net_result.dproc_ssd_result[i].bbox.x_min << " " << ssd_net_result.dproc_ssd_result[i].bbox.y_min << " " << bbox_list.bbox[i].norm_max_x << " " << bbox_list.bbox[i].norm_max_y;
				}

				RVAL_OK(set_overlay_bbox(&bbox_list));
				RVAL_OK(show_overlay(dsp_pts));
#endif

				sum_time += (gettimeus() - start_time);
				++loop_count;
				if (loop_count == TIME_MEASURE_LOOPS) {
					LOG(WARNING) << "det_lpr average time [per " << TIME_MEASURE_LOOPS << " loops]:" << sum_time / (1000 * TIME_MEASURE_LOOPS) << "ms";
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
				LOG(ERROR) << "det_lpr get image fail!";
				continue;
			}
			has_lpr = 0;
			TIME_MEASURE_END("[det_lpr] get yuv cost time", 1);
#elif defined(IS_PC_RUN) && defined(IS_LPR_RUN)
            has_lpr = 0;
			list_index = 0;
			list_has_lpr.bbox_num = 0;
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
		LOG(WARNING) << "det_lpr thread quit.";
	} while (0);

	return NULL;
}

#else

#if defined(USE_OLD_REC)
static void *run_lpr_pthread(void *param_thread)
{
	int rval;
	lpr_thread_params_t *lpr_param =
		(lpr_thread_params_t*)param_thread;
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

	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;

	list_lpr_bbox.bbox_num = 0;

	std::cout << "lpr size:" << lpr_param->width << " " << lpr_param->height << " " << lpr_param->pitch << std::endl;

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
				if(lpr_critical_resource(&license_num, bbox_param, ssd_mid_buf, G_param) < 0)
					continue;
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
				// if (G_param->abort_if_preempted) {
				// 	pthread_mutex_lock(&G_param->vp_access_lock);
				// }
				RVAL_OK(LPR_run_vp_preprocess(&LPR_ctx, img_tensor, license_num, (void*)bbox_param));
				// if (G_param->abort_if_preempted) {
				// 	pthread_mutex_unlock(&G_param->vp_access_lock); // unlock to let SSD run during LPR ARM time
				// }

				RVAL_OK(LPR_run_arm_preprocess(&LPR_ctx, license_num));
				// if (G_param->abort_if_preempted) {
				// 	pthread_mutex_lock(&G_param->vp_access_lock);
				// }
				RVAL_OK(LPR_run_vp_recognition(&LPR_ctx, license_num, &license_result));
#ifdef IS_SHOW
				draw_overlay_preprocess(&draw_plate_list, &license_result, bbox_param, debug_en);
				TIME_MEASURE_START(debug_en);
				RVAL_OK(set_overlay_image(img_tensor, &draw_plate_list));
				TIME_MEASURE_END("[LPR] LPR draw overlay time", debug_en);
#endif
				// if (G_param->abort_if_preempted) {
				// 	pthread_mutex_unlock(&G_param->vp_access_lock);
				// }

				if(license_result.license_num > 0)
				{
					size_t char_len = strlen(license_result.license_info[0].text);
					LOG(INFO) << "LPR:"  << license_result.license_info[0].text << " " \
								<< license_result.license_info[0].conf << " "
								<< char_len;
					if (license_result.license_info[0].conf > DEFAULT_LPR_CONF_THRES && \
						(char_len == 9 || char_len == 10))
						{
							LOG(WARNING) << "LPR: info "  << license_result.license_info[0].text << " " \
							<< license_result.license_info[0].conf;
							pthread_mutex_lock(&result_mutex);
							if(license_result.license_info[0].conf > lpr_confidence)
							{
								lpr_result = license_result.license_info[0].text;
								lpr_confidence = license_result.license_info[0].conf;
								LOG(WARNING) << "LPR:"  << lpr_result << " " << lpr_confidence;
							}
							bbox_param_t lpr_bbox = {0};
							lpr_bbox.norm_min_x = bbox_param[0].norm_min_x * lpr_param->width;
							lpr_bbox.norm_min_y = bbox_param[0].norm_min_y * lpr_param->height;
							lpr_bbox.norm_max_x = bbox_param[0].norm_max_x * lpr_param->width;
							lpr_bbox.norm_max_y = bbox_param[0].norm_max_y * lpr_param->height;
							if(list_lpr_bbox.bbox_num < MAX_OVERLAY_PLATE_NUM)
							{
								list_lpr_bbox.bbox[list_lpr_bbox.bbox_num++] = lpr_bbox;
							}
							else
							{
								LOG(INFO) << "not add lpr bbox";
							}
							LOG(INFO) << "LPR bbox: " << lpr_bbox.norm_min_x << " " \
														<< lpr_bbox.norm_min_y << " " \
														<< lpr_bbox.norm_max_x << " " \
														<< lpr_bbox.norm_max_y;
							pthread_mutex_unlock(&result_mutex);
						}
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
					LOG(WARNING) << "LPR average time:"<< average_time1 << " per license cost time:" << average_time2;
					sum_time = 0;
					loop_count = 1;
					average_license_num = license_num;
				}
			}
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
			lpr_result = "";
			lpr_confidence = 0;
			list_lpr_bbox.bbox_num = 0;
			usleep(20000);
#endif
		}
	} while (0);
	do {
		network_process.send_error(15);
		run_flag = 0;
		free_single_state_buffer(ssd_mid_buf);
		LPR_deinit(&LPR_ctx);
		LOG(WARNING) << "LPR thread quit.";
	} while (0);

	return NULL;
}

#else

static void *run_lpr_pthread(void *param_thread)
{
	int rval;
	unsigned long long int frame_number = 0;
	lpr_thread_params_t *lpr_param =
		(lpr_thread_params_t*)param_thread;
	global_control_param_t *G_param = lpr_param->G_param;
	uint32_t debug_en = G_param->debug_en;

	RecNet rec_net;

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
	
	cv::Mat bgr;
	std::vector<cv::Point2f> polygon;
	float char_score = 0;

	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;

	list_lpr_bbox.bbox_num = 0;

	std::cout << "lpr size:" << lpr_param->width << " " << lpr_param->height << " " << lpr_param->pitch << std::endl;

	prctl(PR_SET_NAME, "lpr_pthread");

	do {
		memset(&license_result, 0, sizeof(license_list_t));
		memset(&draw_plate_list, 0, sizeof(draw_plate_list_t));
		memset(&bbox_param, 0, sizeof(bbox_param));

		RVAL_ASSERT(rec_net.initModel("/data/lpr/mbv3_lstm.bin"));
		RVAL_OK(alloc_single_state_buffer(&G_param->ssd_result_buf, &ssd_mid_buf));

		while (run_flag) {
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
			while(run_lpr > 0)
#endif
			{
				if(lpr_critical_resource(&license_num, bbox_param, ssd_mid_buf, G_param) < 0)
					continue;
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
				// std::stringstream filename_image;
                // filename_image << "/data/save_data/" << "image_" << frame_number << ".jpg";
				// RVAL_OK(ea_tensor_to_jpeg(img_tensor, EA_TENSOR_COLOR_MODE_YUV_NV12, filename_image.str().c_str()));
				// frame_number++;
				TIME_MEASURE_START(debug_en);
				RVAL_OK(tensor2mat_yuv2bgr_nv12(img_tensor, bgr));
				// save_process.save_image(bgr, start_time);
				TIME_MEASURE_END("[LPR] LPR cvt bgr cost time", debug_en);

				TIME_MEASURE_START(debug_en);
				polygon.clear();
				polygon.push_back(cv::Point2f(bbox_param[0].p3_x*bgr.cols, \ 
				                            bbox_param[0].p3_y*bgr.rows));
				polygon.push_back(cv::Point2f(bbox_param[0].p4_x*bgr.cols, \ 
				                            bbox_param[0].p4_y*bgr.rows));
				polygon.push_back(cv::Point2f(bbox_param[0].p1_x*bgr.cols, \ 
				                            bbox_param[0].p1_y*bgr.rows));
				polygon.push_back(cv::Point2f(bbox_param[0].p2_x*bgr.cols, \ 
				                            bbox_param[0].p2_y*bgr.rows));
				cv::Mat roi = rec_net.cropImageROI(bgr, polygon);
				TIME_MEASURE_END("[LPR] LPR crop roi cost time", debug_en);
				// save_process.save_image(roi, start_time);

				TIME_MEASURE_START(debug_en);
				TextLine textLine = rec_net.getTextLine(roi);
				license_result.license_num = 0;
				char_score = 0;
				for (int j = 0; j < textLine.charScores.size(); j++)
				{
					char_score += textLine.charScores[j];
				}
				memset(license_result.license_info[0].text, 0, sizeof(license_result.license_info[0].text));
				snprintf(license_result.license_info[0].text, sizeof(license_result.license_info[0].text),
					"%s", textLine.text.c_str());
				if(textLine.charScores.size() > 0)
					license_result.license_info[0].conf = char_score / textLine.charScores.size();
				else
					license_result.license_info[0].conf = 0;
				++license_result.license_num;

				TIME_MEASURE_END("[LPR] LPR network cost time", debug_en);
				
#ifdef IS_SHOW
				TIME_MEASURE_START(debug_en);
				draw_overlay_preprocess(&draw_plate_list, &license_result, bbox_param, debug_en);
				RVAL_OK(set_overlay_image(img_tensor, &draw_plate_list));
				TIME_MEASURE_END("[LPR] LPR draw overlay time", debug_en);
#endif
				size_t char_len = strlen(license_result.license_info[0].text);
				LOG(INFO) << "LPR:"  << license_result.license_info[0].text << " " \
							<< license_result.license_info[0].conf << " "
							<< char_len << " " << textLine.charScores.size();
				if (license_result.license_info[0].conf > DEFAULT_LPR_CONF_THRES && \
					(char_len == 9 || char_len == 10))
					{
						LOG(WARNING) << "LPR: info "  << license_result.license_info[0].text << " " \
							<< license_result.license_info[0].conf;
						pthread_mutex_lock(&result_mutex);
						if(license_result.license_info[0].conf > lpr_confidence)
						{
							lpr_result = license_result.license_info[0].text;
							lpr_confidence = license_result.license_info[0].conf;
							LOG(WARNING) << "LPR:"  << lpr_result << " " << lpr_confidence;
						}
						bbox_param_t lpr_bbox = {0};
						lpr_bbox.norm_min_x = bbox_param[0].norm_min_x * lpr_param->width;
						lpr_bbox.norm_min_y = bbox_param[0].norm_min_y * lpr_param->height;
						lpr_bbox.norm_max_x = bbox_param[0].norm_max_x * lpr_param->width;
						lpr_bbox.norm_max_y = bbox_param[0].norm_max_y * lpr_param->height;
						if(list_lpr_bbox.bbox_num < MAX_OVERLAY_PLATE_NUM)
						{
							list_lpr_bbox.bbox[list_lpr_bbox.bbox_num++] = lpr_bbox;
						}
						else
						{
							LOG(INFO) << "not add lpr bbox";
						}
						LOG(INFO) << "LPR bbox: " << lpr_bbox.norm_min_x << " " \
													<< lpr_bbox.norm_min_y << " " \
													<< lpr_bbox.norm_max_x << " " \
													<< lpr_bbox.norm_max_y;
						pthread_mutex_unlock(&result_mutex);
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
					LOG(WARNING) << "LPR average time:"<< average_time1 << " per license cost time:" << average_time2;
					sum_time = 0;
					loop_count = 1;
					average_license_num = license_num;
				}
			}
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
			lpr_result = "";
			lpr_confidence = 0;
			list_lpr_bbox.bbox_num = 0;
			usleep(20000);
#endif
		}
	} while (0);
	do {
		network_process.send_error(15);
		run_flag = 0;
		free_single_state_buffer(ssd_mid_buf);
		LOG(WARNING) << "LPR thread quit.";
	} while (0);

	return NULL;
}

#endif

static void *run_det_lpr_pthread(void *thread_params)
{
	int rval = 0;
	unsigned long long int frame_number = 0;
	lpr_thread_params_t *det_lpr_param =
		(lpr_thread_params_t*)thread_params;
	global_control_param_t *G_param = det_lpr_param->G_param;

	// YOLOV5 param
	yolov5_t yolov5_ctx;
	landmark_yolov5_result_s yolov5_net_result;
	int license_box_num = 0;

	state_buffer_t *ssd_mid_buf;
	bbox_list_t bbox_list = {0};

	// image related
	ea_tensor_t *img_tensor = NULL;
	ea_img_resource_data_t data;
	uint32_t dsp_pts = 0;

	// Time measurement
	uint64_t start_time = 0;
	uint64_t debug_time = 0;
	float sum_time = 0.0f;
	uint32_t loop_count = 1;
	uint32_t debug_en = G_param->debug_en;

	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;

	uint32_t list_index = 0;
	list_has_lpr.bbox_num = 0;

#if defined(OFFLINE_DATA)
	ea_tensor_t **tensors = NULL;
	int tensor_num = 1;
	size_t img_shape[4] = {1, 1, det_lpr_param->height * 3 / 2, det_lpr_param->width};
	// void *yuv_data = ea_tensor_data_for_read(dst, EA_CPU);
#endif

	std::cout << "det_lpr size:" << det_lpr_param->width << " " << det_lpr_param->height << " " << det_lpr_param->pitch << std::endl;

	prctl(PR_SET_NAME, "det_lpr_pthread");

	do {
		memset(&data, 0, sizeof(data));
		memset(&bbox_list, 0, sizeof(bbox_list_t));

		memset(&yolov5_ctx, 0, sizeof(yolov5_t));
		memset(&yolov5_net_result, 0, sizeof(landmark_yolov5_result_s));
		RVAL_OK(init_yolov5(&yolov5_ctx, G_param));

		RVAL_OK(alloc_single_state_buffer(&G_param->ssd_result_buf, &ssd_mid_buf));
 
		while (run_flag) {
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
			while(run_lpr > 0)
#endif
			{
				start_time = gettimeus();

				TIME_MEASURE_START(debug_en);
#if defined(OFFLINE_DATA)
				cv::Mat src_image;
				save_process.get_image_yuv(src_image);
				if(src_image.empty())
				{
					LOG(ERROR) << "det_lpr get image fail!";
					continue;
				}
				img_tensor = ea_tensor_new(EA_U8, img_shape, det_lpr_param->pitch);
				mat2tensor_yuv_nv12(src_image, img_tensor);
				tensors = (ea_tensor_t **)malloc(sizeof(ea_tensor_t *) * 1);
				RVAL_ASSERT(tensors != NULL);
				memset(tensors, 0, sizeof(ea_tensor_t *) * 1);
				// led_status = (int *)malloc(sizeof(int) * 1);
				// RVAL_ASSERT(led_status != NULL);
				// memset(led_status, 0, sizeof(int) * 1);
				// led_status[0] = 0;
				data.mono_pts = 0;
				data.dsp_pts = 0;
				data.tensor_group = tensors;
				data.tensor_num = tensor_num;
				data.led_group = NULL;
				data.tensor_group[DEFAULT_SSD_LAYER_ID] = img_tensor;

				// std::stringstream filename_image;
                // filename_image << "/data/save_data/" << "image_" << frame_number << ".jpg";
				// RVAL_OK(ea_tensor_to_jpeg(img_tensor, EA_TENSOR_COLOR_MODE_YUV_NV12, filename_image.str().c_str()));
				// frame_number++;
#else
				RVAL_OK(ea_img_resource_hold_data(G_param->img_resource, &data));
				RVAL_ASSERT(data.tensor_group != NULL);
				RVAL_ASSERT(data.tensor_num >= 1);
				img_tensor = data.tensor_group[DEFAULT_SSD_LAYER_ID];
				dsp_pts = data.dsp_pts;
#endif
				TIME_MEASURE_END("[det_lpr]run_lpr get yuv cost time", debug_en);
				// std::cout << "det_lpr size:" << ea_tensor_shape(img_tensor)[0] << " " << ea_tensor_shape(img_tensor)[1] << " " << ea_tensor_shape(img_tensor)[2] << " " << ea_tensor_shape(img_tensor)[3] << std::endl;
				// SAVE_TENSOR_IN_DEBUG_MODE("det_lpr_pyd.jpg", img_tensor, debug_en);
				// if(frame_number % 80 == 0)
				// {
				// 	has_lpr = 0;
				// }

				TIME_MEASURE_START(debug_en);
				RVAL_OK(ea_cvt_color_resize(img_tensor, yolov5_input(&yolov5_ctx), EA_COLOR_YUV2RGB_NV12, EA_VP));
				TIME_MEASURE_END("[det_lpr] preprocess time", debug_en);

				TIME_MEASURE_START(debug_en);
				rval = yolov5_vp_forward(&yolov5_ctx);
				TIME_MEASURE_END("[det_lpr] network time", debug_en);

				TIME_MEASURE_START(debug_en);
				yolov5_net_result.valid_det_count = 0;
				RVAL_OK(landmark_yolov5_arm_post_process(&yolov5_ctx, &yolov5_net_result));
				TIME_MEASURE_END("[det_lpr] ARM NMS time", debug_en);

				TIME_MEASURE_START(debug_en);
				license_box_num = min(yolov5_net_result.valid_det_count, MAX_DETECTED_LICENSE_NUM);
				bbox_list.bbox_num = min(license_box_num, MAX_OVERLAY_PLATE_NUM);
				RVAL_OK(yolov5_critical_resource(yolov5_net_result.detections, &data, bbox_list.bbox_num, ssd_mid_buf, G_param));

				pthread_mutex_lock(&result_mutex);
				if(bbox_list.bbox_num > 0)
				{
					has_lpr = 1;
				}
				else
				{
					has_lpr = 0;
				}
				if(list_has_lpr.bbox_num < MAX_OVERLAY_PLATE_NUM)
					list_has_lpr.bbox_num++;
				list_has_lpr.has[list_index++] = has_lpr;
				list_index = list_index % MAX_OVERLAY_PLATE_NUM;
				pthread_mutex_unlock(&result_mutex);
				LOG(WARNING) << "lpr box count:" << bbox_list.bbox_num;

				TIME_MEASURE_END("[det_lpr] post-process time", debug_en);

#ifdef IS_SHOW
				for (int i = 0; i < bbox_list.bbox_num; ++i) {
					bbox_list.bbox[i].norm_min_x = yolov5_net_result.detections[i].x_start;
					bbox_list.bbox[i].norm_min_y = yolov5_net_result.detections[i].y_start;
					bbox_list.bbox[i].norm_max_x = yolov5_net_result.detections[i].x_end;
					bbox_list.bbox[i].norm_max_y = yolov5_net_result.detections[i].y_end;
					bbox_list.bbox[i].score = yolov5_net_result.detections[i].score;
					if(yolov5_net_result.detections[i].p1_x > yolov5_net_result.detections[i].p2_x && \
						yolov5_net_result.detections[i].p4_x > yolov5_net_result.detections[i].p3_x)
					{
						bbox_list.bbox[i].p1_x = yolov5_net_result.detections[i].p1_x;
						bbox_list.bbox[i].p1_y = yolov5_net_result.detections[i].p1_y;
						bbox_list.bbox[i].p2_x = yolov5_net_result.detections[i].p2_x;
						bbox_list.bbox[i].p2_y = yolov5_net_result.detections[i].p2_y;
						bbox_list.bbox[i].p3_x = yolov5_net_result.detections[i].p3_x;
						bbox_list.bbox[i].p3_y = yolov5_net_result.detections[i].p3_y;
						bbox_list.bbox[i].p4_x = yolov5_net_result.detections[i].p4_x;
						bbox_list.bbox[i].p4_y = yolov5_net_result.detections[i].p4_y;
					}
					else
					{
						bbox_list.bbox[i].p1_x = yolov5_net_result.detections[i].p2_x;
						bbox_list.bbox[i].p1_y = yolov5_net_result.detections[i].p2_y;
						bbox_list.bbox[i].p2_x = yolov5_net_result.detections[i].p1_x;
						bbox_list.bbox[i].p2_y = yolov5_net_result.detections[i].p1_y;
						bbox_list.bbox[i].p3_x = yolov5_net_result.detections[i].p4_x;
						bbox_list.bbox[i].p3_y = yolov5_net_result.detections[i].p4_y;
						bbox_list.bbox[i].p4_x = yolov5_net_result.detections[i].p3_x;
						bbox_list.bbox[i].p4_y = yolov5_net_result.detections[i].p3_y;
					}

					LOG(INFO) << "lpr norm box:" << yolov5_net_result.detections[i].x_start << " " << \
					                                yolov5_net_result.detections[i].y_start << " " << \
										            yolov5_net_result.detections[i].x_end << " " << \
										            yolov5_net_result.detections[i].y_end << " " << \
													yolov5_net_result.detections[i].p1_x << " " << \
													yolov5_net_result.detections[i].p1_y << " " << \
													yolov5_net_result.detections[i].p2_x << " " << \
													yolov5_net_result.detections[i].p2_y << " " << \
													yolov5_net_result.detections[i].p3_x << " " << \
													yolov5_net_result.detections[i].p3_y << " " << \
													yolov5_net_result.detections[i].p4_x << " " << \
													yolov5_net_result.detections[i].p4_y;
				}

				RVAL_OK(set_overlay_bbox(&bbox_list));
				RVAL_OK(show_overlay(dsp_pts));
#endif

				sum_time += (gettimeus() - start_time);
				++loop_count;
				if (loop_count == TIME_MEASURE_LOOPS) {
					LOG(WARNING) << "det_lpr average time [per " << TIME_MEASURE_LOOPS << " loops]:" << sum_time / (1000 * TIME_MEASURE_LOOPS) << "ms";
					sum_time = 0;
					loop_count = 1;
				}
			}
#if defined(OFFLINE_DATA) && defined(IS_PC_RUN) && defined(IS_LPR_RUN)
            TIME_MEASURE_START(debug_en);
			cv::Mat src_image;
			save_process.get_image(src_image);
			if(src_image.empty())
			{
				LOG(ERROR) << "det_lpr get image fail!";
				continue;
			}
			has_lpr = 0;
			list_index = 0;
			list_has_lpr.bbox_num = 0;
			TIME_MEASURE_END("[det_lpr] get yuv cost time", debug_en);
#elif defined(IS_PC_RUN) && defined(IS_LPR_RUN)
            has_lpr = 0;
			list_index = 0;
			list_has_lpr.bbox_num = 0;
			usleep(20000);
#endif
		}
	} while (0);
	do {
		run_flag = 0;
		network_process.send_error(14);
		yolov5_deinit(&yolov5_ctx);
		free_single_state_buffer(ssd_mid_buf);
		LOG(WARNING) << "det_lpr thread quit.";
	} while (0);

	return NULL;
}

#endif

static void *run_denet_pthread(void *thread_params)
{
	const std::string model_path = "/data/lpr/denet.bin";
	// const std::vector<std::string> input_name = {"data"};
	// const std::vector<std::string> output_name = {"det_output0", "det_output1", "det_output2"};
	const std::vector<std::string> input_name = {"images"};
	const std::vector<std::string> output_name = {"444", "385", "326"};
	const char* class_name[CLASS_NUMBER] = {"car"};

	int rval = 0;
	lpr_thread_params_t *denet_param =
		(lpr_thread_params_t*)thread_params;
	global_control_param_t *G_param = denet_param->G_param;
	uint32_t debug_en = G_param->debug_en;

	std::vector<std::vector<float>> boxes;

	// image related
	ea_tensor_t *img_tensor = NULL;
	ea_img_resource_data_t data;
	uint32_t dsp_pts = 0;
	bbox_list_t bbox_list = {0};
	cv::Mat src_image;
	cv::Mat bgr;
	
	DeNetV2 denet_process;
	// Time measurement
	uint64_t start_time = 0;
	uint64_t debug_time = 0;
	float sum_time = 0.0f;
	uint32_t loop_count = 1;

	int car_count = 0;
	float iou = 0;
#if IMAGE_HEIGHT == 1080
	std::vector<float> roi = {600, 900};
#elif IMAGE_HEIGHT == 720
	std::vector<float> roi = {350, 630};
#endif

	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;

#if defined(OFFLINE_DATA)
	ea_tensor_t **tensors = NULL;
	int tensor_num  = 1;
	size_t img_shape[4] = {1, 1, denet_param->height * 3 / 2, denet_param->width};
#endif

	uint32_t list_index = 0;
	list_has_car.bbox_num = 0;

	std::cout << "denet size:" << denet_param->width << " " << denet_param->height << " " << denet_param->pitch << std::endl;

	prctl(PR_SET_NAME, "run_denet_pthread");

	if(denet_process.init(model_path, input_name, output_name, CLASS_NUMBER, 0.5f) < 0)
    {
		LOG(ERROR) << "DeNet init fail!";
        return NULL;
    }
	memset(&data, 0, sizeof(data));
	denet_process.set_log((int)debug_en);

	while(run_flag)
	{
#if defined(IS_PC_RUN) && defined(IS_CAR_RUN)
		while(run_denet > 0)
#endif
		{
			start_time = gettimeus();
#if defined(OFFLINE_DATA)
			save_process.get_image(src_image);
			if(src_image.empty())
			{
				LOG(ERROR) << "DeNet get image fail!";
				continue;
			}
			save_process.get_image_yuv(src_image);
			
			img_tensor = ea_tensor_new(EA_U8, img_shape, denet_param->pitch);
			mat2tensor_yuv_nv12(src_image, img_tensor);
			tensors = (ea_tensor_t **)malloc(sizeof(ea_tensor_t *) * 1);
			RVAL_ASSERT(tensors != NULL);
			memset(tensors, 0, sizeof(ea_tensor_t *) * 1);
			// led_status = (int *)malloc(sizeof(int) * 1);
			// RVAL_ASSERT(led_status != NULL);
			// memset(led_status, 0, sizeof(int) * 1);
			// led_status[0] = 0;
			data.mono_pts = 0;
			data.dsp_pts = 0;
			data.tensor_group = tensors;
			data.tensor_num = tensor_num;
			data.led_group = NULL;
			data.tensor_group[DEFAULT_LPR_LAYER_ID] = img_tensor;
#else
			// image_geter.get_image(src_image);
			// if(src_image.empty())
			// {
			// 	LOG(ERROR) << "DeNet get image fail!";
			// 	continue;
			// }
			RVAL_OK(ea_img_resource_hold_data(G_param->img_resource, &data));
			RVAL_ASSERT(data.tensor_group != NULL);
			RVAL_ASSERT(data.tensor_num >= 1);
			img_tensor = data.tensor_group[DEFAULT_LPR_LAYER_ID];
			dsp_pts = data.dsp_pts;
			// RVAL_OK(ea_tensor_to_jpeg(img_tensor, EA_TENSOR_COLOR_MODE_YUV_NV12, "image.jpg"));
#endif

#ifdef IS_SAVE
				TIME_MEASURE_START(debug_en);
				RVAL_OK(tensor2mat_yuv2bgr_nv12(img_tensor, bgr));
				save_process.put_image_data(bgr);
				// std::stringstream filename_image;
                // filename_image << save_process.get_image_save_dir() << "image_" << frame_number << ".jpg";
				// RVAL_OK(ea_tensor_to_jpeg(img_tensor, EA_TENSOR_COLOR_MODE_YUV_NV12, filename_image.str().c_str()));
				// TIME_MEASURE_END("yuv to bgr time", debug_en);
#endif
			boxes = denet_process.run(img_tensor);
			car_count = 0;
			bbox_list.bbox_num = min(boxes.size(), MAX_OVERLAY_PLATE_NUM);
			for (size_t i = 0; i < bbox_list.bbox_num; ++i)
			{
				iou = cal_iou(boxes[i], roi);
				LOG(WARNING) << "car iou:" << iou;
				if(iou > 0.01)
				{
					float xmin = boxes[i][0];
					float ymin = boxes[i][1];
					float xmax = xmin + boxes[i][2];
					float ymax = ymin + boxes[i][3];
					int type = boxes[i][4];
					float confidence = boxes[i][5];
					bbox_list.bbox[car_count].norm_min_x = xmin / denet_param->width;
					bbox_list.bbox[car_count].norm_min_y = ymin / denet_param->height;
					bbox_list.bbox[car_count].norm_max_x = xmax / denet_param->width;
					bbox_list.bbox[car_count].norm_max_y = ymax / denet_param->height;
					bbox_list.bbox[car_count].score = confidence;
					car_count++;
					LOG(WARNING) << "car box:" << xmin << " " << ymin << " " << xmax << " " << ymax << " " << confidence;
				}
			}
			
#if defined(IS_PC_RUN) && defined(IS_CAR_RUN)
			pthread_mutex_lock(&result_mutex);
			if(car_count > 0)
			{
				if(list_has_car.bbox_num < MAX_OVERLAY_PLATE_NUM)
					list_has_car.bbox_num++;
				list_has_car.has[list_index++] = 1;
				list_index = list_index % MAX_OVERLAY_PLATE_NUM;
			}
			else
			{
				if(list_has_car.bbox_num < MAX_OVERLAY_PLATE_NUM)
					list_has_car.bbox_num++;
				list_has_car.has[list_index++] = 0;
				list_index = list_index % MAX_OVERLAY_PLATE_NUM;
			}
			pthread_mutex_unlock(&result_mutex);
#endif 
			LOG(WARNING) << "car count:" << car_count;
#ifdef IS_SHOW
			bbox_list.bbox_num = car_count;
			bbox_list.bbox[bbox_list.bbox_num].norm_min_x = roi[0] / denet_param->width;
			bbox_list.bbox[bbox_list.bbox_num].norm_min_y = roi[1] / denet_param->height;
			bbox_list.bbox[bbox_list.bbox_num].norm_max_x = 1;
			bbox_list.bbox[bbox_list.bbox_num].norm_max_y = 1;
			bbox_list.bbox_num++;
			RVAL_OK(set_car_bbox(&bbox_list));
			RVAL_OK(show_overlay(dsp_pts));
#endif

#if defined(OFFLINE_DATA)
			for (int i = 0; i < data.tensor_num; i++) {
				if (data.tensor_group[i]) {
					ea_tensor_free(data.tensor_group[i]);
					data.tensor_group[i] = NULL;
				}
			}
			free(data.tensor_group);
			data.tensor_group = NULL;
			data.led_group = NULL;
#else
			RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &data));
#endif
			sum_time += (gettimeus() - start_time);
			++loop_count;
			if (loop_count == TIME_MEASURE_LOOPS) {
				LOG(WARNING) << "Car det average time [per " << TIME_MEASURE_LOOPS << " loops]:" << sum_time / (1000 * TIME_MEASURE_LOOPS) << "ms";
				sum_time = 0;
				loop_count = 1;
			}
		}
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
		list_index = 0;
	    list_has_car.bbox_num = 0;
		usleep(20000);
#endif
	}
	network_process.send_error(13);
	run_denet = 0;
	LOG(WARNING) << "run_denet_pthread quit！";
	return NULL;
}

static void *process_pc_pthread(void *thread_params)
{
	int rval = 0;
	uint64_t start_time = 0;
	float sum_time = 0.0f;
	float average_license_num = 0.0f;
	uint32_t loop_count = 1;

	global_control_param_t *G_param =
		(global_control_param_t*)thread_params;
	
	uint64_t debug_time = 0;
	uint32_t debug_en = G_param->debug_en;
	bool first_save = true;

	int object_count = 0;
	int bg_point_count = 0;
	unsigned long long int process_number = 0;
	unsigned long long int no_process_number = 0;
	cv::Mat filter_map;
	long pre_stamp = 0;
	long stamp = 0;
	cv::Mat filter_pre_map;
	cv::Mat pre_map;
	cv::Mat bg_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH),CV_8UC1);
	cv::Mat depth_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH),CV_8UC1);
	std::vector<int> point_cout_list;

	cv::Mat img_bgmodel;
	cv::Mat img_output;
	IBGS *bgs = new ViBeBGS();

	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;

#if !defined(IS_LPR_RUN)
	lpr_confidence = 1;
	lpr_result = "12345678";
	has_lpr = 1;
#endif

	point_cout_list.clear();

#if defined(OFFLINE_DATA)
	save_process.get_tof_depth_map(depth_map);
#else
	tof_geter.get_tof_depth_map(depth_map, &stamp);
#endif
	// cv::medianBlur(depth_map, bg_map, 3);
	cv::GaussianBlur(depth_map, bg_map, cv::Size(9, 9), 3.5, 3.5);
	cv::imwrite("./bg.png", bg_map);

	pre_map = depth_map.clone();

	prctl(PR_SET_NAME, "process_pc_pthread");

	while(run_flag > 0)
	{
		start_time = gettimeus();
		TIME_MEASURE_START(debug_en);
#if defined(OFFLINE_DATA)
		save_process.get_tof_depth_map(depth_map);
#else
		rval = tof_geter.get_tof_depth_map(depth_map, &stamp);
		if(rval < 0)
		{
			network_process.send_error(12);
			break;
		}
#endif
		TIME_MEASURE_END("[point_cloud] get TOF cost time", debug_en);

		TIME_MEASURE_START(debug_en);
		cv::GaussianBlur(depth_map, filter_map, cv::Size(9, 9), 3.5, 3.5);
		TIME_MEASURE_END("[point_cloud] filtering cost time", debug_en);

		TIME_MEASURE_START(debug_en);
		bgs->process(filter_map, img_output, img_bgmodel);
		bg_point_count = static_cast<int>(cv::sum(img_output / 255)[0]);
		LOG(WARNING) << "bg_point_count:" << bg_point_count;
		TIME_MEASURE_END("[point_cloud] bgs cost time", debug_en);

		// TIME_MEASURE_START(debug_en);
		// object_count = has_motion_target(img_output);
		// LOG(WARNING) << "motion target count:" << object_count;
		// TIME_MEASURE_END("[point_cloud] bgs cost time", debug_en);

		// if(process_number % 1 == 0)
		// {
		// 	std::stringstream filename;
		// 	filename << "point_cloud" << process_number << ".png";
		// 	cv::imwrite(filename.str(), img_output);
		// 	// dump_bin(filename.str(), src_cloud);
		// }
		// process_number++;

		TIME_MEASURE_START(debug_en);
		if(bg_point_count > 50)
		{
			no_process_number = 0;
			run_lpr = 1;
			run_denet = 1;
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
				save_process.put_tof_data(depth_map, stamp);
#endif
			// if(has_lpr == 1)
			{
				// point_cout_list.push_back(bg_point_count);
				if(point_cout_list.size() == 0)
				{
					cv::GaussianBlur(pre_map, filter_pre_map, cv::Size(9, 9), 3.5, 3.5);
					int point_count = compute_depth_map(bg_map, filter_pre_map);
					point_cout_list.push_back(point_count);
				}
				int point_count = compute_depth_map(bg_map, filter_map);
				point_cout_list.push_back(point_count);
				LOG(WARNING) << "point_count:" << point_count;
			}
		}
		else
		{
			no_process_number++;
			
			// int point_count = compute_depth_map(bg_map, filter_map);
			// point_cout_list.push_back(point_count);
			// LOG(WARNING) << "bg_point_count:" << point_count;
			if(no_process_number % 20 == 0)
			{
				int point_count = compute_depth_map(bg_map, filter_map);
				int in_out_result = vote_in_out(point_cout_list);
				LOG(WARNING) << "final point_count:" << point_count << " " << in_out_result;
				if(in_out_result == 1 && point_count < 100)
				{
					in_out_result = 0;
				}
				else if(in_out_result == 2 && point_count >= 400)
				{
					in_out_result = 0;
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
				run_denet = 0;
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
		TIME_MEASURE_END("[point_cloud] process cost time", debug_en);

		if(process_number % 2 == 0)
		{
			pre_map = depth_map.clone();
			pre_stamp = stamp;
		}
		process_number++;
		
		sum_time += (gettimeus() - start_time);
		++loop_count;
		if (loop_count == TIME_MEASURE_LOOPS) {
			LOG(WARNING) << "PC average time [per " << TIME_MEASURE_LOOPS << " loops]:" << sum_time / (1000 * TIME_MEASURE_LOOPS) << "ms";
			sum_time = 0;
			loop_count = 1;
		}
	}
	network_process.send_error(16);
	run_lpr = 0;
	run_denet = 0;
	run_flag = 0;
	delete bgs;
    bgs = NULL;
	LOG(WARNING) << "process_pc_pthread quit！";
	return NULL;
}

static int start_all(global_control_param_t *G_param)
{
	int rval = 0;

	size_t mb_freedisk = 0;

	pthread_t det_lpr_pthread_id = 0;
	pthread_t lpr_pthread_id = 0;
	pthread_t denet_pthread_id = 0;
	pthread_t process_pc_pthread_id = 0;
	lpr_thread_params_t lpr_thread_params;
	lpr_thread_params_t det_lpr_thread_params;
	lpr_thread_params_t denet_thread_param;

	ea_tensor_t *img_tensor = NULL;
	ea_img_resource_data_t data;

	struct sched_param param;
    pthread_attr_t lpr_attr, det_lpr_attr, denet_attr, pc_attr;
	pthread_attr_init(&lpr_attr);
    pthread_attr_init(&det_lpr_attr);
	pthread_attr_init(&denet_attr);
	pthread_attr_init(&pc_attr);

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

#if defined(IS_PC_RUN)
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
#endif

// #if defined(IS_CAR_RUN)
// 	if(image_geter.start() < 0)
// 	{
// 		rval = -1;
// 		run_flag = 0;
// 		LOG(ERROR) << "start image fail!";
// 		return rval;
// 	}
// 	else
// 	{
// 		LOG(INFO) << "start image success";
// 	}
// #endif

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
		lpr_thread_params.G_param = G_param;
		denet_thread_param.height = ea_tensor_shape(img_tensor)[2];
		denet_thread_param.width = ea_tensor_shape(img_tensor)[3];
		denet_thread_param.pitch = ea_tensor_pitch(img_tensor);
		denet_thread_param.G_param = G_param;
		img_tensor = data.tensor_group[DEFAULT_SSD_LAYER_ID];
		det_lpr_thread_params.height = ea_tensor_shape(img_tensor)[2];
		det_lpr_thread_params.width = ea_tensor_shape(img_tensor)[3];
		det_lpr_thread_params.pitch = ea_tensor_pitch(img_tensor);
		det_lpr_thread_params.G_param = G_param;
		RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &data));
#if defined(IS_LPR_RUN)
		param.sched_priority = 31;
		pthread_attr_setschedpolicy(&det_lpr_attr, SCHED_RR);
		pthread_attr_setschedparam(&det_lpr_attr, &param);
		pthread_attr_setinheritsched(&det_lpr_attr, PTHREAD_EXPLICIT_SCHED);
		rval = pthread_create(&det_lpr_pthread_id, &det_lpr_attr, run_det_lpr_pthread, (void*)&det_lpr_thread_params);
		RVAL_ASSERT(rval == 0);
		param.sched_priority = 31;
		pthread_attr_setschedpolicy(&lpr_attr, SCHED_RR);
		pthread_attr_setschedparam(&lpr_attr, &param);
		pthread_attr_setinheritsched(&lpr_attr, PTHREAD_EXPLICIT_SCHED);
		rval = pthread_create(&lpr_pthread_id, &lpr_attr, run_lpr_pthread, (void*)&lpr_thread_params);
		RVAL_ASSERT(rval == 0);
#endif

#if defined(IS_CAR_RUN)
		param.sched_priority = 31;
		pthread_attr_setschedpolicy(&denet_attr, SCHED_RR);
		pthread_attr_setschedparam(&denet_attr, &param);
		pthread_attr_setinheritsched(&denet_attr, PTHREAD_EXPLICIT_SCHED);
		rval = pthread_create(&denet_pthread_id, &denet_attr, run_denet_pthread, (void*)&denet_thread_param);
		RVAL_ASSERT(rval == 0);
#endif

#if defined(IS_PC_RUN)
		param.sched_priority = 31;
		pthread_attr_setschedpolicy(&pc_attr, SCHED_RR);
		pthread_attr_setschedparam(&pc_attr, &param);
		pthread_attr_setinheritsched(&pc_attr, PTHREAD_EXPLICIT_SCHED);
		rval = pthread_create(&process_pc_pthread_id, &pc_attr, process_pc_pthread, (void*)G_param);
		RVAL_ASSERT(rval == 0);
#endif
	} while (0);
	LOG(INFO) << "start_ssd_lpr success";

	while(run_flag)
	{
		int result = network_process.process_recv();
		mb_freedisk = 0;
		if(get_system_tf_free("/data", &mb_freedisk) == 0)
		{
			LOG(WARNING) << "free disk:" << mb_freedisk << "MB";
			if(mb_freedisk < 10)
			{
				network_process.send_error(18);
			}
		}
		if(result == 200)
		{
			run_lpr = 0;
			run_denet = 0;
			run_flag = 0;
		}
	}
	LOG(WARNING) << "process_recv quit！";

	if (lpr_pthread_id > 0) {
		pthread_join(lpr_pthread_id, NULL);
		lpr_pthread_id = 0;
	}
	LOG(WARNING) << "lpr pthread release";
	if (det_lpr_pthread_id > 0) {
		pthread_join(det_lpr_pthread_id, NULL);
		det_lpr_pthread_id = 0;
	}
	LOG(WARNING) << "det_lpr pthread release";
	if (denet_pthread_id > 0) {
		pthread_join(denet_pthread_id, NULL);
		denet_pthread_id = 0;
	}
	LOG(WARNING) << "denet pthread release";
	if (process_pc_pthread_id > 0) {
		pthread_join(process_pc_pthread_id, NULL);
		process_pc_pthread_id = 0;
	}
	LOG(WARNING) << "PC pthread release";
	pthread_attr_destroy(&lpr_attr);
    pthread_attr_destroy(&det_lpr_attr);
	pthread_attr_destroy(&denet_attr);
	pthread_attr_destroy(&pc_attr);
#if defined(OFFLINE_DATA)
	save_process.offline_stop();
#else
	tof_geter.stop();
// #if defined(IS_CAR_RUN)
// 	image_geter.stop();
// #endif
	save_process.stop();
#endif
	network_process.stop();
	LOG(WARNING) << "process_recv_pthread pthread release";
	pthread_mutex_destroy(&result_mutex);
	LOG(WARNING) << "Main thread quit";
	return rval;
}

#if defined(ONLY_SAVE_DATA)
static void *run_tof_pthread(void *thread_params)
{
	uint64_t debug_time = 0;
	long stamp = 0;
	//unsigned char tof_data[TOF_SIZE];
	cv::Mat depth_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH),CV_8UC1);
	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;
	prctl(PR_SET_NAME, "run_tof_pthread");
	tof_geter.set_up();
	while(run_flag)
	{
		debug_time = gettimeus();
		// tof_geter.get_tof_Z(tof_data);
		// save_process.save_tof_z(depth_map);
		tof_geter.get_tof_depth_map(depth_map, &stamp);
		save_process.save_depth_map(depth_map, stamp);
		LOG(WARNING) << "save tof pthread all cost time:" <<  (gettimeus() - debug_time)/1000.0  << "ms";
	}
	// pthread_detach(pthread_self());
	LOG(WARNING) << "run_tof_pthread quit";
	return NULL;
}

static void *run_image_pthread(void *thread_params)
{
	uint64_t debug_time = 0;
	long stamp = 0;
	// unsigned char yuv_data[IMAGE_YUV_SIZE] = {0};
	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;
	prctl(PR_SET_NAME, "run_image_pthread");
	while(run_flag)
	{
		debug_time = gettimeus();
		cv::Mat src_image; 
		image_geter.get_image(src_image, &stamp);
		// image_geter.get_yuv(yuv_data, &stamp);
		save_process.save_image(src_image, stamp);
		LOG(WARNING) << "save image pthread all cost time:" <<  (gettimeus() - debug_time)/1000.0  << "ms";
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
	run_lpr = 0;
	run_denet = 0;
	run_flag = 0;
	network_process.send_post();
#if defined(OFFLINE_DATA)
	save_process.offline_stop();
#endif
	LOG(WARNING) << "sigstop msg, exit";
}

static void SignalHandle(const char* data, int size) {
    std::string str = data;
	network_process.send_error(19);
	run_lpr = 0;
	run_denet = 0;
	run_flag = 0;
	network_process.send_post();
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
#if defined(ONLY_SAVE_DATA)
	FLAGS_logtostderr = true; //设置为true，就不会写日志文件了
#else
	FLAGS_logtostderr = false;
#endif
	// FLAGS_alsologtostderr = true;
	FLAGS_minloglevel = 0;
	FLAGS_stop_logging_if_full_disk = true;
 
	signal(SIGINT, sigstop);
	signal(SIGQUIT, sigstop);
	signal(SIGTERM, sigstop);
#if defined(ONLY_SAVE_DATA)
    pthread_t tof_pthread_id = 0;
	pthread_t image_pthread_id = 0;
	struct sched_param param;
	pthread_attr_t tof_pthread_attr;
    pthread_attr_t image_pthread_attr;
	pthread_attr_init(&tof_pthread_attr);
    pthread_attr_init(&image_pthread_attr);
	save_process.init_data();
	save_process.init_save_dir();
	if(tof_geter.open_tof() == 0 && image_geter.open_camera() == 0)
	{
		if(start_all() >= 0)
		{
			run_flag = 1;
			param.sched_priority = 21;
			pthread_attr_setschedpolicy(&tof_pthread_attr, SCHED_RR);
			pthread_attr_setschedparam(&tof_pthread_attr, &param);
			pthread_attr_setinheritsched(&tof_pthread_attr, PTHREAD_EXPLICIT_SCHED);
			rval = pthread_create(&tof_pthread_id, &tof_pthread_attr, run_tof_pthread, NULL);
			if(rval >= 0)
			{
				param.sched_priority = 21;
				pthread_attr_setschedpolicy(&image_pthread_attr, SCHED_RR);
				pthread_attr_setschedparam(&image_pthread_attr, &param);
				pthread_attr_setinheritsched(&image_pthread_attr, PTHREAD_EXPLICIT_SCHED);
				rval = pthread_create(&image_pthread_id, &image_pthread_attr, run_image_pthread, NULL);
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
	pthread_attr_destroy(&tof_pthread_attr);
    pthread_attr_destroy(&image_pthread_attr);
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
	save_process.set_save_dir("/data/offline_data/N4/image/", "/data/offline_data/N4/tof/");
	do {
		RVAL_OK(init_param(&G_param));
		RVAL_OK(env_init(&G_param));
		RVAL_OK(start_all(&G_param));
	}while(0);
	env_deinit(&G_param);
#else

#if defined(IS_PC_RUN)
	if(tof_geter.open_tof() == 0 /*&& image_geter.open_camera() == 0*/)
#endif
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


