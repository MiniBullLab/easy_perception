
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
//#define IS_CAR_RUN
//#define IS_PC_RUN

static std::vector<int> list_has_lpr;
static std::vector<int> list_has_car;
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
	int has_car = 0;
	int sum_count = 0;
	float car_sum = 0;
	float lpr_sum = 0;
	std::vector<bbox_param_t> result_bbox;
	if(list_has_car.size() > 0)
	{
		sum_count = 0;
		for (size_t i = list_has_car.size() - 14; i >= 0; i--)
		{
			car_sum += list_has_car[i];
			sum_count++;
			if(sum_count > 30)
			{
				break;
			}
		}
		car_sum = car_sum / sum_count;
		if(lpr_sum > 0.5f)
		{
			has_car = 1;
		}
		else
		{
			has_car = 0;
		}
		LOG(WARNING) << "has_car: " << has_car << " " << car_sum;
	}
	if(lpr_count >= 3)
	{
		sum_count = 0;
		for (size_t i = list_has_lpr.size() - 14; i >= 0; i--)
		{
			lpr_sum += list_has_lpr[i];
			sum_count++;
			if(sum_count > 30)
			{
				break;
			}
		}
		if(sum_count > 10)
		{
			lpr_sum = lpr_sum / sum_count;
			if(lpr_sum > 0.5f)
			{
				lpr_in_out = 1;
			}
			else
			{
				lpr_in_out = 2;
			}
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

	if(send_count >= 1 && final_result == 1 && result_bbox.size() < 2)
	{
		final_result = 0;
	}
	if(send_count >= 1 && lpr_in_out == 1 && result_bbox.size() < 2)
	{
		lpr_in_out = 0;
	}

	if(final_result > 0)
	{
		if(lpr_result != "" && lpr_confidence > 0)
		{
			network_process.send_result(lpr_result, final_result);
			lpr_result = "";
			send_count++;
			lpr_confidence = 0;
		}
		else if(has_car > 0 && final_result == 1)
		{
			network_process.send_result("001", 3);
		}
		else if(has_car > 0 && final_result == 2)
		{
			network_process.send_result("001", 4);
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
		else if(has_car > 0 && lpr_in_out == 1)
		{
			network_process.send_result("001", 3);
		}
		else if(has_car > 0 && lpr_in_out == 2)
		{
			network_process.send_result("001", 4);
		}
		if(final_result == 2)
		{
			send_count = 0;
		}
	}

	list_has_lpr.clear();
	list_lpr_bbox.clear();
	list_has_car.clear();
}

// static void *run_lpr_pthread(void *param_thread)
// {
// 	int rval;
// 	lpr_thread_params_t *lpr_param =
// 		(lpr_thread_params_t*)param_thread;
// 	global_control_param_t *G_param = lpr_param->G_param;
// 	LPR_ctx_t LPR_ctx;

// 	// Detection result param
// 	bbox_param_t bbox_param[MAX_DETECTED_LICENSE_NUM];
// 	draw_plate_list_t draw_plate_list;
// 	uint16_t license_num = 0;
// 	license_list_t license_result;
// 	state_buffer_t *ssd_mid_buf;
// 	ea_img_resource_data_t * data = NULL;
// 	ea_tensor_t *img_tensor = NULL;

// 	// Time mesurement
// 	uint64_t start_time = 0;
// 	uint64_t debug_time = 0;
// 	float sum_time = 0.0f;
// 	float average_license_num = 0.0f;
// 	uint32_t loop_count = 1;
// 	uint32_t debug_en = G_param->debug_en;

// 	prctl(PR_SET_NAME, "lpr_pthread");

// 	do {
// 		memset(&LPR_ctx, 0, sizeof(LPR_ctx));
// 		memset(&draw_plate_list, 0, sizeof(draw_plate_list));
// 		memset(&bbox_param, 0, sizeof(bbox_param));

// 		LPR_ctx.img_h = lpr_param->height;
// 		LPR_ctx.img_w = lpr_param->width;
// 		RVAL_OK(init_LPR(&LPR_ctx, G_param));
// 		RVAL_OK(alloc_single_state_buffer(&G_param->ssd_result_buf, &ssd_mid_buf));

// 		while (run_flag) {
// #if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
// 			while(run_lpr > 0)
// #endif
// 			{
// 				if(lpr_critical_resource(&license_num, bbox_param, ssd_mid_buf, G_param) < 0)
// 					continue;
// 				start_time = gettimeus();
// 				data = (ea_img_resource_data_t *)ssd_mid_buf->img_resource_addr;
// 				if (license_num == 0) {
// #if defined(OFFLINE_DATA)
// 					for (int i = 0; i < data->tensor_num; i++) {
// 						if (data->tensor_group[i]) {
// 							ea_tensor_free(data->tensor_group[i]);
// 							data->tensor_group[i] = NULL;
// 						}
// 					}
// 					free(data->tensor_group);
// 					data->tensor_group = NULL;
// 					data->led_group = NULL;
// #else
// 					RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, data));
// #endif
// 					continue;
// 				}
// 				img_tensor = data->tensor_group[DEFAULT_LPR_LAYER_ID];
// 				if (G_param->abort_if_preempted) {
// 					pthread_mutex_lock(&G_param->vp_access_lock);
// 				}
// 				RVAL_OK(LPR_run_vp_preprocess(&LPR_ctx, img_tensor, license_num, (void*)bbox_param));
// 				if (G_param->abort_if_preempted) {
// 					pthread_mutex_unlock(&G_param->vp_access_lock); // unlock to let SSD run during LPR ARM time
// 				}

// 				RVAL_OK(LPR_run_arm_preprocess(&LPR_ctx, license_num));
// 				if (G_param->abort_if_preempted) {
// 					pthread_mutex_lock(&G_param->vp_access_lock);
// 				}
// 				RVAL_OK(LPR_run_vp_recognition(&LPR_ctx, license_num, &license_result));
// #ifdef IS_SHOW
// 				draw_overlay_preprocess(&draw_plate_list, &license_result, bbox_param, G_param);
// 				TIME_MEASURE_START(debug_en);
// 				RVAL_OK(set_overlay_image(img_tensor, &draw_plate_list));
// 				TIME_MEASURE_END("[LPR] LPR draw overlay time", debug_en);
// #endif
// 				if (G_param->abort_if_preempted) {
// 					pthread_mutex_unlock(&G_param->vp_access_lock);
// 				}

// 				if(license_result.license_num > 0)
// 				{
// 					pthread_mutex_lock(&result_mutex);
// 					bbox_param_t lpr_bbox = {0};
// 					lpr_bbox.norm_min_x = bbox_param[0].norm_min_x * LPR_ctx.img_w;
// 					lpr_bbox.norm_min_y = bbox_param[0].norm_min_y * LPR_ctx.img_h;
// 					lpr_bbox.norm_max_x = bbox_param[0].norm_max_x * LPR_ctx.img_w;
// 					lpr_bbox.norm_max_y = bbox_param[0].norm_max_y * LPR_ctx.img_h;

// #if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
// 					list_lpr_bbox.push_back(lpr_bbox);
// #endif

// 					LOG(INFO) << "lpr bbox: " << lpr_bbox.norm_min_x << " " \
// 					<< lpr_bbox.norm_min_y << " " \
// 				    << lpr_bbox.norm_max_x << " " \
// 					<< lpr_bbox.norm_max_y;
					
// 					LOG(INFO) << "LPR:"  << license_result.license_info[0].text << " " << license_result.license_info[0].conf;
// 					if (license_result.license_info[0].conf > DEFAULT_LPR_CONF_THRES && \
// 						strlen(license_result.license_info[0].text) == CHINESE_LICENSE_STR_LEN && \
// 						license_result.license_info[0].conf > lpr_confidence)
// 						{
// 							lpr_result = license_result.license_info[0].text;
// 							lpr_confidence = license_result.license_info[0].conf;
// 							LOG(WARNING) << "LPR:"  << lpr_result << " " << lpr_confidence;
// 						}
// 					pthread_mutex_unlock(&result_mutex);
// 				}
// 				else
// 				{

// 				}

// #if defined(OFFLINE_DATA)
// 				for (int i = 0; i < data->tensor_num; i++) {
// 					if (data->tensor_group[i]) {
// 						ea_tensor_free(data->tensor_group[i]);
// 						data->tensor_group[i] = NULL;
// 					}
// 				}
// 				free(data->tensor_group);
// 				data->tensor_group = NULL;
// 				data->led_group = NULL;
// #else
// 				RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, data));
// #endif
// 				sum_time += (gettimeus() - start_time);
// 				++loop_count;
// 				average_license_num += license_num;
// 				if (loop_count == TIME_MEASURE_LOOPS) {
// 					float average_time1 = sum_time / (1000 * TIME_MEASURE_LOOPS);
// 					float average_time2 = (average_license_num > 0.0f) ? (sum_time / (1000 * average_license_num)) : 0.0f;
// 					LOG(INFO) << "[" << TIME_MEASURE_LOOPS  << "loops] LPR average time license_num " << " " << average_license_num / TIME_MEASURE_LOOPS;
// 					LOG(WARNING) << "LPR average time:"<< average_time1 << " per license cost time:" << average_time2;
// 					sum_time = 0;
// 					loop_count = 1;
// 					average_license_num = license_num;
// 				}
// 			}
// 			usleep(20000);
// 		}
// 	} while (0);
// 	do {
// 		run_flag = 0;
// 		free_single_state_buffer(ssd_mid_buf);
// 		LPR_deinit(&LPR_ctx);
// 		LOG(WARNING) << "LPR thread quit.";
// 	} while (0);

// 	return NULL;
// }

static void *run_lpr_pthread(void *param_thread)
{
	int rval;
	lpr_thread_params_t *lpr_param =
		(lpr_thread_params_t*)param_thread;
	global_control_param_t *G_param = lpr_param->G_param;

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
	uint32_t debug_en = G_param->debug_en;

	cv::Mat bgr(lpr_param->height * 2 / 3, lpr_param->width, CV_8UC3);
	std::vector<cv::Mat> imgs;
	std::vector<cv::Point> polygon;
	float char_score = 0;

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
				RVAL_OK(tensor2mat_yuv2bgr_nv12(img_tensor, bgr));
				imgs.clear();
				polygon.clear();
				polygon.push_back(cv::Point(static_cast<int>(bbox_param[0].p3_x*bgr.cols), \ 
				                            static_cast<int>(bbox_param[0].p3_y*bgr.rows)));
				polygon.push_back(cv::Point(static_cast<int>(bbox_param[0].p4_x*bgr.cols), \ 
				                            static_cast<int>(bbox_param[0].p4_y*bgr.rows)));
				polygon.push_back(cv::Point(static_cast<int>(bbox_param[0].p1_x*bgr.cols), \ 
				                            static_cast<int>(bbox_param[0].p1_y*bgr.rows)));
				polygon.push_back(cv::Point(static_cast<int>(bbox_param[0].p2_x*bgr.cols), \ 
				                            static_cast<int>(bbox_param[0].p2_y*bgr.rows)));
				cv::Mat roi = rec_net.cropImageROI(bgr, polygon);
				imgs.push_back(roi);
				// save_process.save_image(roi, start_time);
				std::vector<TextLine> textLines = rec_net.getTextLines(imgs);
				license_result.license_num = 0;
				char_score = 0;
				for (int i = 0; i < textLines.size(); i++)
				{
					TextLine text = textLines[i];
					for (int j = 0; j < text.charScores.size(); j++)
					{
						char_score += text.charScores[j];
					}

					memset(license_result.license_info[i].text, 0, sizeof(license_result.license_info[i].text));
					snprintf(license_result.license_info[i].text, sizeof(license_result.license_info[i].text),
						"%s", text.text.c_str());
					license_result.license_info[i].conf = char_score / text.charScores.size();
					++license_result.license_num;
				}
				
#ifdef IS_SHOW
				draw_overlay_preprocess(&draw_plate_list, &license_result, bbox_param, G_param);
				TIME_MEASURE_START(debug_en);
				RVAL_OK(set_overlay_image(img_tensor, &draw_plate_list));
				TIME_MEASURE_END("[LPR] LPR draw overlay time", debug_en);
#endif

				if(license_result.license_num > 0)
				{
					pthread_mutex_lock(&result_mutex);
					bbox_param_t lpr_bbox = {0};
					lpr_bbox.norm_min_x = bbox_param[0].norm_min_x * lpr_param->width;
					lpr_bbox.norm_min_y = bbox_param[0].norm_min_y * lpr_param->height;
					lpr_bbox.norm_max_x = bbox_param[0].norm_max_x * lpr_param->width;
					lpr_bbox.norm_max_y = bbox_param[0].norm_max_y * lpr_param->height;

#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
					list_lpr_bbox.push_back(lpr_bbox);
#endif

					LOG(INFO) << "lpr bbox: " << lpr_bbox.norm_min_x << " " \
					<< lpr_bbox.norm_min_y << " " \
				    << lpr_bbox.norm_max_x << " " \
					<< lpr_bbox.norm_max_y;
					
					LOG(INFO) << "LPR:"  << license_result.license_info[0].text << " " << license_result.license_info[0].conf;
					if (license_result.license_info[0].conf > DEFAULT_LPR_CONF_THRES && \
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
					LOG(WARNING) << "LPR average time:"<< average_time1 << " per license cost time:" << average_time2;
					sum_time = 0;
					loop_count = 1;
					average_license_num = license_num;
				}
			}
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
			usleep(20000);
#endif
		}
	} while (0);
	do {
		run_flag = 0;
		free_single_state_buffer(ssd_mid_buf);
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
	// SSD_ctx_t SSD_ctx;
	// ssd_net_final_result_t ssd_net_result;
	// int ssd_result_num = 0;
	// bbox_param_t scaled_license_plate;

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

#if defined(OFFLINE_DATA)
	ea_tensor_t **tensors = NULL;
	int tensor_num;
	size_t img_shape[4] = {1, 1, det_lpr_param->height * 3 / 2, det_lpr_param->width};
	// void *yuv_data = ea_tensor_data_for_read(dst, EA_CPU);
#endif


	prctl(PR_SET_NAME, "ssd_pthread");

	do {
		memset(&data, 0, sizeof(data));
		memset(&bbox_list, 0, sizeof(bbox_list_t));

		// memset(&SSD_ctx, 0, sizeof(SSD_ctx_t));
		// memset(&ssd_net_result, 0, sizeof(ssd_net_result));
		// memset(&scaled_license_plate, 0, sizeof(scaled_license_plate));
		// RVAL_OK(init_ssd(&SSD_ctx, G_param, det_lpr_param->height, det_lpr_param->width));
		// ssd_net_result.dproc_ssd_result = (dproc_ssd_detection_output_result_t *)
		// 	malloc(SSD_ctx.vp_result_info.max_dproc_ssd_result_num *
		// 	sizeof(dproc_ssd_detection_output_result_t));
		// RVAL_ASSERT(ssd_net_result.dproc_ssd_result != NULL);

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
				// RVAL_OK(ea_cvt_color_resize(img_tensor, SSD_ctx.net_input.tensor, EA_COLOR_YUV2BGR_NV12, EA_VP));
				RVAL_OK(ea_cvt_color_resize(img_tensor, yolov5_input(&yolov5_ctx), EA_COLOR_YUV2BGR_NV12, EA_VP));
				TIME_MEASURE_END("[det_lpr] preprocess time", debug_en);

				TIME_MEASURE_START(debug_en);
				// if (G_param->abort_if_preempted) {
				// 	wait_vp_available(&G_param->vp_access_lock);
				// }
				// rval = ssd_net_run_vp_forward(&SSD_ctx.ssd_net_ctx);
				rval = yolov5_vp_forward(&yolov5_ctx);
				// if (rval < 0) {
				// 	if (rval == -EAGAIN) {
				// 		do {
				// 			if (G_param->abort_if_preempted) {
				// 				wait_vp_available(&G_param->vp_access_lock);
				// 			}
				// 			rval = ea_net_resume(yolov5_ctx.net);
				// 			if (rval == -EINTR) {
				// 				printf("det_lpr network interrupt by signal in resume\n");
				// 				break;
				// 			}
				// 		} while (rval == -EAGAIN);
				// 	} else if (rval == -EINTR) {
				// 		printf("det_lpr network interrupt by signal in run\n");
				// 		break;
				// 	} else {
				// 		printf("ssd_net_run_vp_forward failed, ret: %d\n", rval);
				// 		rval = -1;
				// 		break;
				// 	}
				// }
			
				// ea_tensor_sync_cache(SSD_ctx.ssd_net_ctx.output_loc_tensor, EA_VP, EA_CPU);
				// ea_tensor_sync_cache(SSD_ctx.ssd_net_ctx.output_conf_tensor, EA_VP, EA_CPU);
				TIME_MEASURE_END("[det_lpr] network time", debug_en);

				TIME_MEASURE_START(debug_en);
				// ssd_net_result.ssd_det_num = 0;
				// memset(&ssd_net_result.labels[0][0], 0,
				// 	SSD_NET_MAX_LABEL_NUM * SSD_NET_MAX_LABEL_LEN);
				// memset(ssd_net_result.dproc_ssd_result, 0,
				// 	SSD_ctx.vp_result_info.max_dproc_ssd_result_num *
				// 	sizeof(dproc_ssd_detection_output_result_t));
				// RVAL_OK(ssd_net_run_arm_nms(&SSD_ctx.ssd_net_ctx,
				// 	SSD_ctx.vp_result_info.loc_dram_addr,
				// 	SSD_ctx.vp_result_info.conf_dram_addr, &ssd_net_result));
				
				yolov5_net_result.valid_det_count = 0;
				RVAL_OK(landmark_yolov5_arm_post_process(&yolov5_ctx, &yolov5_net_result));
				TIME_MEASURE_END("[det_lpr] ARM NMS time", debug_en);

				TIME_MEASURE_START(debug_en);
				// ssd_result_num = min(ssd_net_result.ssd_det_num, MAX_DETECTED_LICENSE_NUM);
				// bbox_list.bbox_num = min(ssd_result_num, MAX_OVERLAY_PLATE_NUM);
				// ssd_critical_resource(ssd_net_result.dproc_ssd_result, &data,
				// 	bbox_list.bbox_num, ssd_mid_buf, G_param);

				license_box_num = min(yolov5_net_result.valid_det_count, MAX_DETECTED_LICENSE_NUM);
				bbox_list.bbox_num = min(license_box_num, MAX_OVERLAY_PLATE_NUM);
				yolov5_critical_resource(yolov5_net_result.detections, &data, bbox_list.bbox_num, ssd_mid_buf, G_param);

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

				TIME_MEASURE_END("[det_lpr] post-process time", debug_en);

#ifdef IS_SHOW
				// for (int i = 0; i < bbox_list.bbox_num; ++i) {
				// 	upscale_normalized_rectangle(ssd_net_result.dproc_ssd_result[i].bbox.x_min,
				// 	ssd_net_result.dproc_ssd_result[i].bbox.y_min,
				// 	ssd_net_result.dproc_ssd_result[i].bbox.x_max,
				// 	ssd_net_result.dproc_ssd_result[i].bbox.y_max,
				// 	DRAW_LICNESE_UPSCALE_W, DRAW_LICNESE_UPSCALE_H, &scaled_license_plate);
				// 	bbox_list.bbox[i].norm_min_x = scaled_license_plate.norm_min_x;
				// 	bbox_list.bbox[i].norm_min_y = scaled_license_plate.norm_min_y;
				// 	bbox_list.bbox[i].norm_max_x = scaled_license_plate.norm_max_x;
				// 	bbox_list.bbox[i].norm_max_y = scaled_license_plate.norm_max_y;

				// 	LOG(INFO) << "lpr :" << ssd_net_result.dproc_ssd_result[i].bbox.x_min << " " << ssd_net_result.dproc_ssd_result[i].bbox.y_min << " " << bbox_list.bbox[i].norm_max_x << " " << bbox_list.bbox[i].norm_max_y;
				// }

				for (int i = 0; i < bbox_list.bbox_num; ++i) {
					bbox_list.bbox[i].norm_min_x = yolov5_net_result.detections[i].x_start;
					bbox_list.bbox[i].norm_min_y = yolov5_net_result.detections[i].y_start;
					bbox_list.bbox[i].norm_max_x = yolov5_net_result.detections[i].x_end;
					bbox_list.bbox[i].norm_max_y = yolov5_net_result.detections[i].y_end;
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
										            yolov5_net_result.detections[i].y_end;
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
			usleep(20000);
#endif
		}
	} while (0);
	do {
		run_flag = 0;
		// if (ssd_net_result.dproc_ssd_result != NULL) {
		// 	free(ssd_net_result.dproc_ssd_result);
		// }
		// ssd_net_deinit(&SSD_ctx.ssd_net_ctx);
		yolov5_deinit(&yolov5_ctx);
		free_single_state_buffer(ssd_mid_buf);
		LOG(WARNING) << "det_lpr thread quit.";
	} while (0);

	return NULL;
}

static void *run_denet_pthread(void *thread_params)
{
	const std::string model_path = "/data/lpr/denet.bin";
	// const std::vector<std::string> input_name = {"data"};
	// const std::vector<std::string> output_name = {"det_output0", "det_output1", "det_output2"};
	const std::vector<std::string> input_name = {"images"};
	const std::vector<std::string> output_name = {"444", "385", "326"};
	const char* class_name[CLASS_NUMBER] = {"car"};
	global_control_param_t *G_param = (global_control_param_t*)thread_params;
	int rval = 0;
	std::vector<std::vector<float>> boxes;
	// image related
	ea_tensor_t *img_tensor = NULL;
	ea_img_resource_data_t data;
	uint32_t dsp_pts = 0;
	bbox_list_t bbox_list = {0};
	cv::Mat src_image;
    // DeNet denet_process;
	DeNetV2 denet_process;
	// Time measurement
	uint64_t start_time = 0;
	uint64_t debug_time = 0;
	float sum_time = 0.0f;
	uint32_t loop_count = 1;

	int width = 0;
	int height = 0;

	int car_count = 0;
	float iou = 0;
	std::vector<float> roi = {600, 900, 1320, 180};

	// cv::Mat bgr(height * 2 / 3, width, CV_8UC3);

	prctl(PR_SET_NAME, "run_denet_pthread");

	if(denet_process.init(model_path, input_name, output_name, CLASS_NUMBER, 0.5f) < 0)
    {
		LOG(ERROR) << "DeNet init fail!";
        return NULL;
    }
	memset(&data, 0, sizeof(data));

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
			img_tensor = data.tensor_group[0];
			dsp_pts = data.dsp_pts;
			width = ea_tensor_shape(img_tensor)[3];
			height = ea_tensor_shape(img_tensor)[2];
			// RVAL_OK(ea_tensor_to_jpeg(img_tensor, EA_TENSOR_COLOR_MODE_YUV_NV12, "image.jpg"));
#endif

#ifdef IS_SAVE
				TIME_MEASURE_START(debug_en);
				// RVAL_OK(tensor2mat_yuv2bgr_nv12(img_tensor, bgr));
				// save_process.put_image_data(bgr);
				std::stringstream filename_image;
                filename_image << save_process.get_image_save_dir() << "image_" << frame_number << ".jpg";
				RVAL_OK(ea_tensor_to_jpeg(img_tensor, EA_TENSOR_COLOR_MODE_YUV_NV12, filename_image.str().c_str()));
				TIME_MEASURE_END("yuv to bgr time", debug_en);
#endif
			boxes = denet_process.run(img_tensor);
			car_count = 0;
			bbox_list.bbox_num = min(boxes.size(), MAX_OVERLAY_PLATE_NUM);
			for (size_t i = 0; i < bbox_list.bbox_num; ++i)
			{
				iou = cal_iou(boxes[i], roi);
				LOG(WARNING) << "car iou:" << iou;
				if(iou > 0.2f)
				{
					float xmin = boxes[i][0];
					float ymin = boxes[i][1];
					float xmax = xmin + boxes[i][2];
					float ymax = ymin + boxes[i][3];
					int type = boxes[i][4];
					float confidence = boxes[i][5];
					bbox_list.bbox[car_count].norm_min_x = xmin / width;
					bbox_list.bbox[car_count].norm_min_y = ymin / height;
					bbox_list.bbox[car_count].norm_max_x = xmax / width;
					bbox_list.bbox[car_count].norm_max_y = ymax / height;
					car_count++;
					LOG(WARNING) << "car box:" << xmin << " " << ymin << " " << xmax << " " << ymax << " " << confidence;
				}
			}
			bbox_list.bbox_num = car_count;
#if defined(IS_PC_RUN) && defined(IS_CAR_RUN)
			if(bbox_list.bbox_num > 0)
			{
				pthread_mutex_lock(&result_mutex);
				list_has_car.push_back(1);
				pthread_mutex_unlock(&result_mutex);
			}
			else
			{
				pthread_mutex_lock(&result_mutex);
				list_has_car.push_back(0);
				pthread_mutex_unlock(&result_mutex);
			}
#endif 
			LOG(WARNING) << "car count:" <<bbox_list.bbox_num;
#ifdef IS_SHOW
			RVAL_OK(set_car_bbox(&bbox_list));
			RVAL_OK(show_overlay(dsp_pts));
#endif
			RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &data));
			sum_time += (gettimeus() - start_time);
			++loop_count;
			if (loop_count == TIME_MEASURE_LOOPS) {
				LOG(WARNING) << "Car det average time [per " << TIME_MEASURE_LOOPS << " loops]:" << sum_time / (1000 * TIME_MEASURE_LOOPS) << "ms";
				sum_time = 0;
				loop_count = 1;
			}
		}
#if defined(IS_PC_RUN) && defined(IS_LPR_RUN)
		usleep(50000);
#endif
	}
	network_process.send_error(12);
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
			run_lpr = 0;
			run_denet = 0;
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

	int object_count = 0;
	int bg_point_count = 0;
	unsigned long long int process_number = 0;
	unsigned long long int no_process_number = 0;
	cv::Mat filter_map;
	long pre_stamp = 0;
	long stamp = 0;
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
	tof_geter.get_tof_depth_map(depth_map, &stamp);
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
		tof_geter.get_tof_depth_map(depth_map, &stamp);
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

		TIME_MEASURE_START(debug_en);
		object_count = has_motion_target(img_output);
		LOG(WARNING) << "motion target count:" << object_count;
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
				point_cout_list.push_back(bg_point_count);
			}
		}
		else
		{
			no_process_number++;
			if(no_process_number % 15 == 0)
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
		TIME_MEASURE_END("[point_cloud] cost time", debug_en);

		if(process_number % 10 == 0)
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
	run_lpr = 0;
	run_denet = 0;
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
	pthread_t det_lpr_pthread_id = 0;
	pthread_t lpr_pthread_id = 0;
	lpr_thread_params_t lpr_thread_params;
	lpr_thread_params_t det_lpr_thread_params;

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
		img_tensor = data.tensor_group[DEFAULT_SSD_LAYER_ID];
		det_lpr_thread_params.height = ea_tensor_shape(img_tensor)[2];
		det_lpr_thread_params.width = ea_tensor_shape(img_tensor)[3];
		det_lpr_thread_params.pitch = ea_tensor_pitch(img_tensor);
		det_lpr_thread_params.G_param = G_param;
		RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &data));
#if defined(IS_LPR_RUN)
		rval = pthread_create(&det_lpr_pthread_id, NULL, run_det_lpr_pthread, (void*)&det_lpr_thread_params);
		RVAL_ASSERT(rval == 0);
		rval = pthread_create(&lpr_pthread_id, NULL, run_lpr_pthread, (void*)&lpr_thread_params);
		RVAL_ASSERT(rval == 0);
#endif

#if defined(IS_CAR_RUN)
		rval = pthread_create(&denet_pthread_id, NULL, run_denet_pthread, (void*)G_param);
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
	long stamp = 0;
	//unsigned char tof_data[TOF_SIZE];
	cv::Mat depth_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH),CV_8UC1);
	int policy;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
        printf("SCHED_OTHER\n");
    if(policy == SCHED_RR)
    	printf("SCHED_RR \n");
    if(policy == SCHED_FIFO)
        printf("SCHED_FIFO\n");
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
	int policy;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
	std::cout << "policy:" << policy << std::endl;
    if(policy == SCHED_OTHER)
        printf("SCHED_OTHER\n");
    if(policy == SCHED_RR)
    	printf("SCHED_RR\n");
    if(policy == SCHED_FIFO)
        printf("SCHED_FIFO\n");
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
#if defined(OFFLINE_DATA)
	save_process.offline_stop();
#endif
	LOG(WARNING) << "sigstop msg, exit";
}

static void SignalHandle(const char* data, int size) {
    std::string str = data;
	network_process.send_error(18);
	run_lpr = 0;
	run_denet = 0;
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
	save_process.set_save_dir("/data/offline_data/2/image/", "/data/offline_data/2/tof/");
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


