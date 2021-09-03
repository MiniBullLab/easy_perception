#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <signal.h>
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <iav_ioctl.h>
#include <eazyai.h>
#include <tensor_private.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "lib_data_process.h"
#include "cnn_lpr/lpr/utils.hpp"
#include "cnn_lpr/lpr/ssd_lpr_common.h"
#include "cnn_lpr/lpr/state_buffer.h"
#include "cnn_lpr/ssd/ssd.h"
#include "cnn_lpr/lpr/overlay_tool.h"
#include "cnn_lpr/lpr/lpr.hpp"

#include "cnn_lpr/tof/tof_acquisition.h"
#include "cnn_lpr/tof/tof_data_process.h"
#include "cnn_lpr/tof/vibebgs.h"

#include "utility/utils.h"

#define FILENAME_LENGTH				(256)
#define MAX_NET_NUM					(16)
#define DEFAULT_STATE_BUF_NUM		(3)
#define DEFAULT_STREAM_ID			(2)
#define DEFAULT_CHANNEL_ID			(2)
#define DEFAULT_SSD_LAYER_ID		(1)
#define DEFAULT_LPR_LAYER_ID		(0)
#define DEFAULT_RGB_TYPE			(1) /* 0: RGB, 1:BGR */
#define DEFAULT_SSD_CLASS_NUM		(2) /* For license and background */
#define DEFAULT_BACKGROUND_ID		(0)
#define DEFAULT_KEEP_TOP_K			(50)
#define DEFAULT_NMS_TOP_K			(100)
#define DEFAULT_NMS_THRES			(0.45f)
#define DEFAULT_SSD_CONF_THRES		(0.3f)
#define DEFAULT_LPR_CONF_THRES		(0.9f)
#define DRAW_LICNESE_UPSCALE_W		(0.2f)
#define DRAW_LICNESE_UPSCALE_H		(1.0f)
#define CHINESE_LICENSE_STR_LEN		(9)
#define TIME_MEASURE_LOOPS			(100)

#define DEPTH_WIDTH (240)
#define DEPTH_HEIGTH (180)

#define IMAGE_BUFFER_SIZE (40)

EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);

const static std::string ssd_model_path = "./lpr/mobilenetv1_ssd_cavalry.bin";
const static std::string ssd_priorbox_path = "./lpr/lpr_priorbox_fp32.bin";
const static std::vector<std::string> ssd_input_name = {"data"};
const static std::vector<std::string> ssd_output_name = {"mbox_loc", "mbox_conf_flatten"};

const static std::string lpr_model_path = "./lpr/segfree_inception_cavalry.bin";
const static std::vector<std::string> lpr_input_name = {"data"};
const static std::vector<std::string> lpr_output_name = {"prob"};

const static std::string lphm_model_path = "./lpr/LPHM_cavalry.bin";
const static std::vector<std::string> lphm_input_name = {"data"};
const static std::vector<std::string> lphm_output_name = {"dense"};

const static std::string bg_point_cloud_file = "./bg.png";

static float lpr_confidence = 0;
static std::string lpr_result = "";

volatile int has_lpr = 0;
static pthread_mutex_t result_mutex;
static pthread_mutex_t ssd_mutex;

struct ImageBuffer  
{  	
	cv::Mat buffer[IMAGE_BUFFER_SIZE];
    pthread_mutex_t lock; /* 互斥体lock 用于对缓冲区的互斥操作 */  
    int readpos, writepos; /* 读写指针*/  
    pthread_cond_t notempty; /* 缓冲区非空的条件变量 */  
    pthread_cond_t notfull; /* 缓冲区未满的条件变量 */  
};

typedef struct global_control_param_s {
	// cmd line param
	uint8_t channel_id;
	uint8_t stream_id;
	uint8_t ssd_pyd_idx;
	uint8_t lpr_pyd_idx;
	uint32_t num_classes;
	uint32_t state_buf_num;
	uint32_t background_label_id;
	uint32_t keep_top_k;
	uint32_t nms_top_k;
	float nms_threshold;
	float conf_threshold;
	float recg_threshold;
	float overlay_text_width_ratio;
	float overlay_x_offset;
	uint16_t overlay_highlight_sec;
	uint16_t overlay_clear_sec;
	uint32_t verbose; /* network print time */
	uint32_t debug_en; /* 0: disable; 1: time measure,2: log; 3: run once & save picture */
	uint32_t rgb_type; /* 0: RGB; 1: BGR */
	uint32_t draw_plate_num;
	ea_img_resource_t *img_resource;

	// run time control
	state_buffer_param_t ssd_result_buf;
	pthread_mutex_t access_buffer_mutex;
	sem_t sem_readable_buf;
} global_control_param_t;

typedef struct SSD_ctx_s {
	ea_net_t *net;
	ssd_net_ctx_t ssd_net_ctx;
	ssd_net_input_t net_input;
	ssd_net_vp_result_info_t vp_result_info;
} SSD_ctx_t;

typedef struct lpr_thread_params_s {
	uint16_t width;
	uint16_t height;
	uint16_t pitch;
	uint16_t reserved;
	global_control_param_t *G_param;
} ssd_lpr_thread_params_t;

enum cavalry_priotiry {
	SSD_PRIORITY,
	VPROC_PRIORITY,
	LPR_PRIORITY,
	PRIORITY_NUM
};

volatile int run_flag = 1;
volatile int run_lpr = 0;

TOFAcquisition tof_geter;
struct ImageBuffer image_buffer;

static int init_param(global_control_param_t *G_param)
{
	int rval = 0;
	std::cout << "init ..." << std::endl;
	memset(G_param, 0, sizeof(global_control_param_t));

	G_param->channel_id = DEFAULT_CHANNEL_ID;
	G_param->stream_id = DEFAULT_STREAM_ID;

	G_param->ssd_pyd_idx= DEFAULT_SSD_LAYER_ID;
	G_param->lpr_pyd_idx= DEFAULT_LPR_LAYER_ID;
	G_param->rgb_type = DEFAULT_RGB_TYPE;
	G_param->state_buf_num = DEFAULT_STATE_BUF_NUM;
	G_param->num_classes = DEFAULT_SSD_CLASS_NUM;
	G_param->background_label_id = DEFAULT_BACKGROUND_ID;
	G_param->keep_top_k = DEFAULT_KEEP_TOP_K;
	G_param->nms_top_k = DEFAULT_NMS_TOP_K;
	G_param->nms_threshold = DEFAULT_NMS_THRES;
	G_param->conf_threshold = DEFAULT_SSD_CONF_THRES;
	G_param->recg_threshold = DEFAULT_LPR_CONF_THRES;
	G_param->overlay_x_offset = DEFAULT_X_OFFSET;
	G_param->overlay_highlight_sec = DEFAULT_HIGHLIGHT_SEC;
	G_param->overlay_clear_sec = DEFAULT_CLEAR_SEC;
	G_param->overlay_text_width_ratio = DEFAULT_WIDTH_RATIO;
	G_param->draw_plate_num = DEFAULT_OVERLAY_LICENSE_NUM;
	G_param->debug_en = 0;
	G_param->verbose = 0;

	std::cout << "init sucess" << std::endl;

	return rval;
}

static int tensor2mat_yuv2bgr_nv12(ea_tensor_t *tensor, cv::Mat &bgr)
{
	int rval = EA_SUCCESS;
	cv::Mat nv12(ea_tensor_shape(tensor)[2] +
		(ea_tensor_related(tensor) == NULL ? 0 : ea_tensor_shape(ea_tensor_related(tensor))[2]),
		ea_tensor_shape(tensor)[3], CV_8UC1);
	uint8_t *p_src = NULL;
	uint8_t *p_dst = NULL;
	size_t h;

	do {
		RVAL_ASSERT(ea_tensor_shape(tensor)[1] == 1);

		p_src = (uint8_t *)ea_tensor_data_for_read(tensor, EA_CPU);
		p_dst = nv12.data;
		for (h = 0; h < ea_tensor_shape(tensor)[2]; h++) {
			memcpy(p_dst, p_src, ea_tensor_shape(tensor)[3]);
			p_src += ea_tensor_pitch(tensor);
			p_dst += ea_tensor_shape(tensor)[3];
		}

		if (ea_tensor_related(tensor)) {
			p_src = (uint8_t *)ea_tensor_data_for_read(ea_tensor_related(tensor), EA_CPU);
			for (h = 0; h < ea_tensor_shape(ea_tensor_related(tensor))[2]; h++) {
				memcpy(p_dst, p_src, ea_tensor_shape(ea_tensor_related(tensor))[3]);
				p_src += ea_tensor_pitch(ea_tensor_related(tensor));
				p_dst += ea_tensor_shape(ea_tensor_related(tensor))[3];
			}
		}

		#if CV_VERSION_MAJOR < 4
			cv::cvtColor(nv12, bgr, CV_YUV2BGR_NV12);
		#else
			cv::cvtColor(nv12, bgr, COLOR_YUV2BGR_NV12);
		#endif
	} while (0);

	return rval;
}

static void upscale_normalized_rectangle(float x_min, float y_min,
	float x_max, float y_max, float w_ratio, float h_ratio,
	bbox_param_t *bbox_scaled)
{
	float obj_h = y_max - y_min;
	float obj_w = x_max - x_min;

	bbox_scaled->norm_min_x = max(0.0f, (x_min - obj_w * w_ratio / 2));
	bbox_scaled->norm_min_y = max(0.0f, (y_min - obj_h * h_ratio / 2));
	bbox_scaled->norm_max_x = min(1.0f, x_min + obj_w * (1.0f + w_ratio / 2));
	bbox_scaled->norm_max_y = min(1.0f, y_min + obj_h * (1.0f + h_ratio / 2));

	return;
}

static void draw_overlay_preprocess(draw_plate_list_t *draw_plate_list,
	license_list_t *license_result, bbox_param_t *bbox_param, global_control_param_t *G_param)
{
	uint32_t i = 0;
	int draw_num = 0;
	bbox_param_t scaled_bbox_draw[MAX_DETECTED_LICENSE_NUM];
	license_plate_t *plates = draw_plate_list->license_plate;
	license_info_t *license_info = license_result->license_info;

	license_result->license_num = min(license_result->license_num, MAX_OVERLAY_PLATE_NUM);
	for (i = 0; i < license_result->license_num; ++i) {
		if (license_info[i].conf > G_param->recg_threshold &&
			strlen(license_info[i].text) == CHINESE_LICENSE_STR_LEN) {
			upscale_normalized_rectangle(bbox_param[i].norm_min_x, bbox_param[i].norm_min_y,
			bbox_param[i].norm_max_x, bbox_param[i].norm_max_y,
				DRAW_LICNESE_UPSCALE_W, DRAW_LICNESE_UPSCALE_H, &scaled_bbox_draw[i]);
			plates[draw_num].bbox.norm_min_x = scaled_bbox_draw[i].norm_min_x;
			plates[draw_num].bbox.norm_min_y = scaled_bbox_draw[i].norm_min_y;
			plates[draw_num].bbox.norm_max_x = scaled_bbox_draw[i].norm_max_x;
			plates[draw_num].bbox.norm_max_y = scaled_bbox_draw[i].norm_max_y;
			plates[draw_num].conf = license_info[i].conf;
			memset(plates[draw_num].text, 0, sizeof(plates[draw_num].text));
			strncpy(plates[draw_num].text, license_info[i].text,
				sizeof(plates[draw_num].text));
			plates[draw_num].text[sizeof(plates[draw_num].text) - 1] = '\0';
			++draw_num;
			if (G_param->debug_en > 0) {
				printf("********************************************************\n");
				printf("\nDrawed license: %s, conf: %f\n\n",
					license_info[i].text, license_info[i].conf);
				printf("********************************************************\n");
			}
		}
	}
	draw_plate_list->license_num = draw_num;

	return;
}

static int lpr_critical_resource(uint16_t *license_num, bbox_param_t *bbox_param,
	state_buffer_t *ssd_mid_buf, global_control_param_t *G_param)
{
	int i, rval = 0;

	do {
		RVAL_OK(read_state_buffer(ssd_mid_buf, &G_param->ssd_result_buf,
			&G_param->access_buffer_mutex, &G_param->sem_readable_buf));
		*license_num = ssd_mid_buf->object_num;
		if (*license_num > MAX_DETECTED_LICENSE_NUM) {
			printf("ERROR: license_num[%d] > MAX_DETECTED_LICENSE_NUM[%d].\n",
				*license_num, MAX_DETECTED_LICENSE_NUM);
			rval = -1;
			break;
		}
		for (i = 0; i < *license_num; ++i) {
			bbox_param[i].norm_min_x = ssd_mid_buf->bbox_param[i].norm_min_x;
			bbox_param[i].norm_min_y = ssd_mid_buf->bbox_param[i].norm_min_y;
			bbox_param[i].norm_max_x = ssd_mid_buf->bbox_param[i].norm_max_x;
			bbox_param[i].norm_max_y = ssd_mid_buf->bbox_param[i].norm_max_y;
		}
		if (G_param->debug_en >= INFO_LEVEL) {
			printf("\n-----------------------------------------------------------------------\n");
			printf("LPR got bboxes:\n");
			for (i = 0; i < *license_num; ++i) {
				printf("%d\t(%4f, %4f)\t\t(%4f, %4f)\n", i,
					bbox_param[i].norm_min_x, bbox_param[i].norm_min_y,
					bbox_param[i].norm_max_x, bbox_param[i].norm_max_y);
			}
			printf("-----------------------------------------------------------------------\n\n");
		}
	} while (0);

	return rval;
}

static int init_LPR(LPR_ctx_t *LPR_ctx, global_control_param_t *G_param)
{
	int rval = 0;
	std::cout << "init_LPR" << std::endl;
	do {
		LPR_ctx->LPHM_net_ctx.input_name = const_cast<char*>(lphm_input_name[0].c_str());
	    LPR_ctx->LPHM_net_ctx.output_name = const_cast<char*>(lphm_output_name[0].c_str());
		LPR_ctx->LPHM_net_ctx.net_name = const_cast<char*>(lphm_model_path.c_str());
		LPR_ctx->LPHM_net_ctx.net_verbose = G_param->verbose;
		LPR_ctx->LPHM_net_ctx.net_param.priority = LPR_PRIORITY;

		LPR_ctx->LPR_net_ctx.input_name = const_cast<char*>(lpr_input_name[0].c_str());
		LPR_ctx->LPR_net_ctx.output_name = const_cast<char*>(lpr_output_name[0].c_str());
		LPR_ctx->LPR_net_ctx.net_name = const_cast<char*>(lpr_model_path.c_str());
		LPR_ctx->LPR_net_ctx.net_verbose = G_param->verbose;
		LPR_ctx->LPR_net_ctx.net_param.priority = LPR_PRIORITY;

		LPR_ctx->debug_en = G_param->debug_en;
		RVAL_OK(LPR_init(LPR_ctx));
	} while (0);
	std::cout << "init_LPR success" << std::endl;
	return rval;
}

static void deinit_LPR(LPR_ctx_t *LPR_ctx)
{
	LPR_deinit(LPR_ctx);
	EA_LOG_NOTICE("LPR_deinit.\n");

	return;
}

// static int put_image_buffer(cv::Mat &image_mat)
// {
// 	uint64_t start_time = 0;
// 	int rval = EA_SUCCESS;
// 	if(image_mat.empty())
// 	{
// 		std::cout << "image empty!" << std::endl;
// 		return -1;
// 	}
// 	pthread_mutex_lock(&image_buffer.lock);  
//     if ((image_buffer.writepos + 1) % IMAGE_BUFFER_SIZE == image_buffer.readpos)  
//     {  
//         pthread_cond_wait(&image_buffer.notfull, &image_buffer.lock);  
//     }
// 	image_buffer.buffer[image_buffer.writepos] = image_mat.clone();
//     image_buffer.writepos++;  
//     if (image_buffer.writepos >= IMAGE_BUFFER_SIZE)  
//         image_buffer.writepos = 0;  
//     pthread_cond_signal(&image_buffer.notempty);  
//     pthread_mutex_unlock(&image_buffer.lock);  
// 	std::cout << "put image" << std::endl;
// 	return rval;
// }

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

	do {
		memset(&LPR_ctx, 0, sizeof(LPR_ctx));
		memset(&draw_plate_list, 0, sizeof(draw_plate_list));
		memset(&bbox_param, 0, sizeof(bbox_param));

		LPR_ctx.img_h = lpr_param->height;
		LPR_ctx.img_w = lpr_param->width;
		RVAL_OK(init_LPR(&LPR_ctx, G_param));
		RVAL_OK(alloc_single_state_buffer(&G_param->ssd_result_buf, &ssd_mid_buf));

		while (run_flag) {
			while(run_lpr > 0)
			{
				RVAL_OK(lpr_critical_resource(&license_num, bbox_param,
				ssd_mid_buf, G_param));
				start_time = gettimeus();
				data = (ea_img_resource_data_t *)ssd_mid_buf->img_resource_addr;
				if (license_num == 0) {
					RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, data));
					continue;
				}
				img_tensor = data->tensor_group[G_param->lpr_pyd_idx];
				RVAL_OK(LPR_run(&LPR_ctx, img_tensor, license_num,
					(void*)bbox_param, &license_result));
				draw_overlay_preprocess(&draw_plate_list, &license_result,
					bbox_param, G_param);
				if(license_result.license_num > 0)
				{
					pthread_mutex_lock(&result_mutex);
					if (license_result.license_info[0].conf > G_param->recg_threshold && \
						strlen(license_result.license_info[0].text) == CHINESE_LICENSE_STR_LEN && \
						license_result.license_info[0].conf > lpr_confidence)
						{
							lpr_result = license_result.license_info[0].text;
							lpr_confidence = license_result.license_info[0].conf;
							std::cout << "LPR:"  << lpr_result << " " << lpr_confidence << std::endl;
						}
					pthread_mutex_unlock(&result_mutex);
				}
				TIME_MEASURE_START(debug_en);
				RVAL_OK(set_overlay_image(img_tensor, &draw_plate_list));
				TIME_MEASURE_END("[LPR] LPR draw overlay time", debug_en);
				RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, data));
				sum_time += (gettimeus() - start_time);
				++loop_count;
				average_license_num += license_num;
				if (loop_count == TIME_MEASURE_LOOPS) {
					printf("[%d loops] LPR average time license_num[%f]: %f ms, per license %f ms\n",
						TIME_MEASURE_LOOPS, average_license_num / TIME_MEASURE_LOOPS,
						sum_time / (1000 * TIME_MEASURE_LOOPS),
						((average_license_num > 0.0f) ? (sum_time / (1000 * average_license_num)) : 0.0f));
					sum_time = 0;
					loop_count = 1;
					average_license_num = license_num;
				}
				if (debug_en == DEBUG_LEVEL) {
					run_flag = 0;
					printf("In debug mode, stop after one loop!\n");
				}
			}
			usleep(20000);
		}
	} while (0);
	do {
		run_flag = 0;
		free_single_state_buffer(ssd_mid_buf);
		deinit_LPR(&LPR_ctx);
		printf("LPR thread quit.\n");
	} while (0);

	return NULL;
}

static int ssd_critical_resource(
	dproc_ssd_detection_output_result_t *amba_ssd_result,
	ea_img_resource_data_t* imgs_data_addr, int ssd_result_num,
	state_buffer_t *ssd_mid_buf, global_control_param_t *G_param)
{
	int i, rval = 0;
	ea_img_resource_data_t covered_imgs_addr;
	uint8_t buffer_covered = 0;

	do {
		ssd_result_num = min(MAX_DETECTED_LICENSE_NUM, ssd_result_num);
		for (i = 0; i < ssd_result_num; i++) {
			ssd_mid_buf->bbox_param[i].norm_max_x =
				amba_ssd_result[i].bbox.x_max;
			ssd_mid_buf->bbox_param[i].norm_max_y =
				amba_ssd_result[i].bbox.y_max;
			ssd_mid_buf->bbox_param[i].norm_min_x =
				amba_ssd_result[i].bbox.x_min;
			ssd_mid_buf->bbox_param[i].norm_min_y =
				amba_ssd_result[i].bbox.y_min;
		}
		ssd_mid_buf->object_num = ssd_result_num;
		memcpy(ssd_mid_buf->img_resource_addr, imgs_data_addr,
			G_param->ssd_result_buf.img_resource_len);
		RVAL_OK(write_state_buffer(&G_param->ssd_result_buf, ssd_mid_buf,
			&G_param->access_buffer_mutex, &G_param->sem_readable_buf,
			(void*)&covered_imgs_addr, &buffer_covered));
		if (buffer_covered) {
			RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &covered_imgs_addr));
		}
	} while (0);

	return rval;
}

static int init_ssd(SSD_ctx_t *SSD_ctx, global_control_param_t *G_param,
	uint32_t buffer_h, uint32_t buffer_w)
{
	int rval = 0;
	ssd_net_params_t ssd_net_params;
	ssd_tf_scale_factors_t scale_factors;
	std::cout << "init_ssd" << std::endl;
	do {
		memset(&ssd_net_params, 0, sizeof(ssd_net_params));
		// set params for ssd_net
		ssd_net_params.model_path = ssd_model_path.c_str();
		ssd_net_params.priorbox_path = ssd_priorbox_path.c_str();
		ssd_net_params.label_path = NULL;
		ssd_net_params.input_name = ssd_input_name[0].c_str();
		ssd_net_params.output_loc = ssd_output_name[0].c_str();
		ssd_net_params.output_conf = ssd_output_name[1].c_str();
		ssd_net_params.width = buffer_w;
		ssd_net_params.height = buffer_h;
		ssd_net_params.conf_threshold = G_param->conf_threshold;
		ssd_net_params.keep_top_k = G_param->keep_top_k;
		ssd_net_params.nms_threshold = G_param->nms_threshold;
		ssd_net_params.nms_top_k = G_param->nms_top_k;
		ssd_net_params.background_label_id = G_param->background_label_id;
		ssd_net_params.unnormalized = 0;
		ssd_net_params.class_num = G_param->num_classes;
		ssd_net_params.priority = SSD_PRIORITY;
		ssd_net_params.debug_en = (G_param->debug_en >= INFO_LEVEL);
		ssd_net_params.nnctrl_print_time = (G_param->verbose);
		scale_factors.center_x_scale = 0;
		scale_factors.center_y_scale = 0;
		scale_factors.height_scale = 0;
		scale_factors.width_scale = 0;
		ssd_net_params.scale_factors = &scale_factors;
		RVAL_OK(ssd_net_init(&ssd_net_params, &SSD_ctx->ssd_net_ctx,
			&SSD_ctx->net_input, &SSD_ctx->vp_result_info));
	} while (0);
	std::cout << "init_ssd success" << std::endl;
	return rval;
}

static void deinit_ssd(SSD_ctx_t *SSD_ctx)
{
	ssd_net_deinit(&SSD_ctx->ssd_net_ctx);
	EA_LOG_NOTICE("deinit_ssd\n");
}

static int led_process(const cv::Mat &bgr)
{
	cv::Mat gray;
	cv::cvtColor(bgr, gray, CV_BGR2GRAY);
	//std::cout << "image size:" << gray.cols << " " << gray.rows << std::endl;
	cv::Scalar left_mean = cv::mean(gray(cv::Rect(0, 0, 300, 300)));  
	cv::Scalar right_mean = cv::mean(gray(cv::Rect(gray.cols-1-300, 0, 300, 300)));
	//std::cout << "left:" << left_mean.val[0] << " right:" << right_mean.val[0] << std::endl;
	if(left_mean.val[0] < 50 && right_mean.val[0] < 50)
	{
		return 1;
	}
	else
	{
		return 0;
	} 
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
	uint32_t dsp_pts;

	// Time measurement
	uint64_t start_time = 0;
	uint64_t debug_time = 0;
	float sum_time = 0.0f;
	uint32_t loop_count = 1;
	uint32_t debug_en = G_param->debug_en;

	bool first_save = true;
	// cv::VideoWriter output_video;
	cv::Mat bgr(ssd_param->height * 2 / 3, ssd_param->width, CV_8UC3);
	struct timeval tv;  
    char time_str[64];

	int is_night = 0;
	int led_device = open("/sys/devices/platform/e4000000.n_apb/e4008000.i2c/i2c-0/0-0064/leds/lm36011:torch/brightness", O_RDWR, 0);
	if (led_device < 0) {
		printf("open led fail\n");
        return NULL;
	}

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
			while(run_lpr > 0){
				RVAL_OK(ea_img_resource_hold_data(G_param->img_resource, &data));
				RVAL_ASSERT(data.tensor_group != NULL);
				RVAL_ASSERT(data.tensor_num >= 1);
				img_tensor = data.tensor_group[G_param->ssd_pyd_idx];
				dsp_pts = data.dsp_pts;
				// SAVE_TENSOR_IN_DEBUG_MODE("SSD_pyd.jpg", img_tensor, debug_en);
				if(frame_number % 80 == 0)
				{
					// std::stringstream filename;
					// filename << "image_" << frame_number << ".jpg";
					// std::cout << "tensor channel:" << ea_tensor_shape(img_tensor)[1] << std::endl;
					// RVAL_OK(ea_tensor_to_jpeg(img_tensor, EA_TENSOR_COLOR_MODE_YUV_NV12, filename.str().c_str()));
					has_lpr = 0;
				}
				frame_number++;

				start_time = gettimeus();

				TIME_MEASURE_START(debug_en);
				RVAL_OK(tensor2mat_yuv2bgr_nv12(img_tensor, bgr));
				TIME_MEASURE_END("[SSD] yuv to bgr time", debug_en);

				if(led_device > 0)
				{
					TIME_MEASURE_START(debug_en);
					is_night = led_process(bgr);
					if(is_night > 0)
					{
						// std::cout << "is_night:" << is_night << std::endl;
						write(led_device, "20", sizeof(char));
					}
					else
					{
						write(led_device, "0", sizeof(char));
					}
					TIME_MEASURE_END("[SSD] led time", debug_en);
				}

				TIME_MEASURE_START(debug_en);
				RVAL_OK(ea_cvt_color_resize(img_tensor, SSD_ctx.net_input.tensor,
					EA_COLOR_YUV2BGR_NV12, EA_VP));
				TIME_MEASURE_END("[SSD] preprocess time", debug_en);

				TIME_MEASURE_START(debug_en);
				RVAL_OK(ssd_net_run_vp_forward(&SSD_ctx.ssd_net_ctx));
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

					has_lpr = 1;
				}

				RVAL_OK(set_overlay_bbox(&bbox_list));
				RVAL_OK(show_overlay(dsp_pts));
				TIME_MEASURE_END("[SSD] post-process time", debug_en);

				// if(has_lpr > 0 && first_save)
				// {
				// 	if(output_video.isOpened())
				// 	{
				// 		output_video.release();
				// 	}
				// 	std::stringstream filename;
				// 	gettimeofday(&tv, NULL);  
				// 	strftime(time_str, sizeof(time_str)-1, "%Y-%m-%d_%H:%M:%S", localtime(&tv.tv_sec)); 
				// 	filename << "./result_video/" << time_str << ".avi";
				// 	if(output_video.open(filename.str(), cv::VideoWriter::fourcc('X','V','I','D'), 25, \
				// 		cv::Size(bgr.cols, bgr.rows)))
				// 	{
				// 		std::cout << "open video save fail!" << std::endl;
				// 		first_save = false;
				// 	}
				// }
				// else if(has_lpr > 0)
				// {
				// 	if (output_video.isOpened())
				// 	{
				// 		output_video.write(bgr);
				// 	}
				// }
				// else
				// {
				// 	if(output_video.isOpened())
				// 	{
				// 		output_video.release();
				// 		first_save = true;
				// 	}
				// }

				sum_time += (gettimeus() - start_time);
				++loop_count;
				if (loop_count == TIME_MEASURE_LOOPS) {
					printf("SSD average time [per %d loops]: %f ms\n",
						TIME_MEASURE_LOOPS, sum_time / (1000 * TIME_MEASURE_LOOPS));
					sum_time = 0;
					loop_count = 1;
				}
			}
			// if (output_video.isOpened())
			// {
			// 	output_video.release();
			// 	first_save = true;
			// }
			write(led_device, "0", sizeof(char));
			has_lpr = 0;
			usleep(20000);
		}
	} while (0);
	do {
		run_flag = 0;
		if (ssd_net_result.dproc_ssd_result != NULL) {
			free(ssd_net_result.dproc_ssd_result);
		}
		deinit_ssd(&SSD_ctx);
		free_single_state_buffer(ssd_mid_buf);
		printf("SSD thread quit.\n");
	} while (0);

	if(led_device >= 0)
    {
		write(led_device, "0", sizeof(char));
        close(led_device);
		led_device = -1;
		printf("close led\n");
    }

	// if(output_video.isOpened())
    // {
    //     output_video.release();
    // }

	return NULL;
}

// static void *save_video_pthread(void* save_data)
// {
// 	int rval = 0;
// 	bool first_save = true;
// 	unsigned long long int frame_number = 0;
// 	cv::Mat image_mat;
// 	cv::VideoWriter output_video;
// 	struct timeval tv;  
//     char time_str[64]; 
// 	while (run_flag > 0) 
// 	{
// 		while(run_lpr > 0)
// 		{
// 			pthread_mutex_lock(&image_buffer.lock);  
// 			if (image_buffer.writepos == image_buffer.readpos)  
// 			{  
// 				pthread_cond_wait(&image_buffer.notempty, &image_buffer.lock);  
// 			}
// 			image_mat = image_buffer.buffer[image_buffer.readpos];
// 			image_buffer.readpos++;  
// 			if (image_buffer.readpos >= IMAGE_BUFFER_SIZE)  
// 				image_buffer.readpos = 0; 
// 			pthread_cond_signal(&image_buffer.notfull);  
// 			pthread_mutex_unlock(&image_buffer.lock);
// 			if(has_lpr > 0 && first_save)
// 			{
// 				if (output_video.isOpened())
// 				{
// 					output_video.release();
// 				}
// 				std::stringstream filename;
// 				gettimeofday(&tv, NULL);  
// 				strftime(time_str, sizeof(time_str)-1, "%Y-%m-%d_%H:%M:%S", localtime(&tv.tv_sec)); 
// 				filename << "./result_video/" << time_str << ".avi";
// 				if(output_video.open(filename.str(), cv::VideoWriter::fourcc('X','V','I','D'), 25, \
// 					cv::Size(image_mat.cols, image_mat.rows)))
// 				{
// 					first_save = false;
// 				}
// 			}
// 			else if(has_lpr > 0)
// 			{
// 				if (output_video.isOpened())
// 				{
// 					output_video.write(image_mat);
// 				}
// 			}
// 			else
// 			{
// 				if (output_video.isOpened())
// 				{
// 					output_video.release();
// 					first_save = true;
// 				}
// 			}
// 			std::cout << "save image" << std::endl;
// 		}
// 		if (output_video.isOpened())
// 		{
// 			output_video.release();
// 			first_save = true;
// 		}
// 		usleep(20000);
// 		// std::cout << "save video runing" << std::endl;
// 	}
// 	if(output_video.isOpened())
//     {
//         output_video.release();
//     }
// 	std::cout << "stop save video pthread" << std::endl;
// 	return NULL;
// }

// static void offline_point_cloud_process()
// {
// 	std::string cloud_dir = "./point_cloud/";
// 	std::vector<std::string> cloud_files;
// 	int point_count = 0;
// 	unsigned long long int frame_number = 0;
// 	std::vector<int> point_cout_list;
// 	TOFAcquisition::PointCloud src_cloud;
// 	TOFAcquisition::PointCloud bg_cloud;
// 	point_cout_list.clear();
// 	read_bin(bg_point_cloud_file, bg_cloud);
// 	ListImages(cloud_dir, cloud_files);
//     std::cout << "total Test cloud : " << cloud_files.size() << std::endl;
// 	while(run_flag > 0)
// 	{
// 		for (size_t index = 0; index < cloud_files.size(); index++) {
// 			std::stringstream temp_str;
//         	temp_str << cloud_dir << cloud_files[index];
// 			std::cout << temp_str.str() << std::endl;
// 			read_bin(temp_str.str(), src_cloud);
// 			if(src_cloud.size() > 10)
// 			{
// 				run_lpr = 1;
// 				if(frame_number % 10 == 0)
// 				{
// 					vote_in_out(point_cout_list);
// 					point_cout_list.clear();
// 				}
// 				point_count = compute_point_count(bg_cloud, src_cloud);
// 				std::cout << "point count:" << point_count << std::endl;
// 				if(point_count > 10)
// 					point_cout_list.push_back(point_count);
// 			}
// 		}
// 	}
// }

static void point_cloud_process(const global_control_param_t *G_param, const int *udp_socket_fd)
{
	uint64_t debug_time = 0;
	uint32_t debug_en = G_param->debug_en;
	int bg_point_count = 0;
	// int is_in = -1;
	// int point_count = 0;
	unsigned long long int process_number = 0;
	unsigned long long int no_process_number = 0;
	cv::Mat filter_map;
	cv::Mat bg_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH),CV_8UC1);
	cv::Mat depth_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH),CV_8UC1);
	std::vector<int> result_list;
	std::vector<int> point_cout_list;
	TOFAcquisition::PointCloud src_cloud;

	cv::Mat img_bgmodel;
	cv::Mat img_output;
	IBGS *bgs = new ViBeBGS();

	struct timeval tv;  
    char time_str[64];
	int dest_port = 9998;
	struct sockaddr_in dest_addr = {0};
    dest_addr.sin_family = AF_INET;
	dest_addr.sin_port = htons(dest_port);
	dest_addr.sin_addr.s_addr = inet_addr("10.0.0.102");

	result_list.clear();
	point_cout_list.clear();

	tof_geter.get_tof_data(src_cloud, depth_map);
	// cv::medianBlur(depth_map, bg_map, 3);
	cv::GaussianBlur(depth_map, bg_map, cv::Size(9, 9), 3.5, 3.5);

	while(run_flag > 0)
	{
		TIME_MEASURE_START(debug_en);
		tof_geter.get_tof_data(src_cloud, depth_map);
		TIME_MEASURE_END("[point_cloud] get TOF cost time", debug_en);

		TIME_MEASURE_START(debug_en);
		// cv::medianBlur(depth_map, filter_map, 3);
		cv::GaussianBlur(depth_map, filter_map, cv::Size(9, 9), 3.5, 3.5);
		TIME_MEASURE_END("[point_cloud] filtering cost time", debug_en);

		TIME_MEASURE_START(debug_en);
		bgs->process(filter_map, img_output, img_bgmodel);
		bg_point_count = static_cast<int>(cv::sum(img_output / 255)[0]);
		std::cout << "bg_point_count:" << bg_point_count << std::endl;
		TIME_MEASURE_END("[point_cloud] bgs cost time", debug_en);

		//point_count = compute_depth_map(bg_map, filter_map);
		//std::cout << "point_count:" << point_count << std::endl;

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
			tof_geter.set_up();
			run_lpr = 1;
			if(has_lpr == 1)
			{
				point_cout_list.push_back(bg_point_count);
				// process_number++;
				// if(process_number % 10 == 0)
				// {
				// 	is_in = vote_in_out(point_cout_list);
				// 	result_list.push_back(is_in);
				// 	point_cout_list.clear();
				// }
			}
		}
		if(bg_point_count <= 50 || has_lpr == 0)
		{
			no_process_number++;
			if(no_process_number % 10 == 0)
			{
				//int final_result = get_in_out(result_list);
				int final_result = vote_in_out(point_cout_list);
				std::cout << "final_result:" << final_result << std::endl;
				pthread_mutex_lock(&result_mutex);
				if(final_result >= 0)
				{
					if(lpr_result != "" && lpr_confidence > 0)
					{
						std::stringstream send_result;
						gettimeofday(&tv, NULL);  
						strftime(time_str, sizeof(time_str)-1, "%Y-%m-%d_%H:%M:%S", localtime(&tv.tv_sec)); 
						send_result << time_str << "|" << final_result << "|" << lpr_result;
						sendto(*udp_socket_fd, send_result.str().c_str(), strlen(send_result.str().c_str()), \
							0, (struct sockaddr *)&dest_addr,sizeof(dest_addr));
						std::cout << send_result.str() << std::endl;
						lpr_result = "";
						lpr_confidence = 0;
					}
				}
				pthread_mutex_unlock(&result_mutex);
				point_cout_list.clear();
				result_list.clear();
				process_number = 0;
				no_process_number = 0;
				has_lpr = 0;
				run_lpr = 0;
				tof_geter.set_sleep();
				bg_map = filter_map.clone();
			}
		}
		else
		{
			no_process_number = 0;
		}
		TIME_MEASURE_END("[point_cloud] cost time", debug_en);
	}
	delete bgs;
    bgs = NULL;
	std::cout << "stop point cloud process" << std::endl;
}

static void* upd_broadcast_send(void* save_data)
{
	int rval = 0;
	int broadcast_port = 8888;
	int on = 1; //开启
	struct sockaddr_in broadcast_addr = {0};
	char buf[1024] = "LPR Runing!";
	int broadcast_socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
	if (broadcast_socket_fd == -1)
    {
        printf("create socket failed ! error message :%s\n", strerror(errno));
        return NULL;
    }
	//开启发送广播数据功能
	rval = setsockopt(broadcast_socket_fd, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on));
	if(rval < 0)
	{
		perror("setsockopt fail\n");
		return NULL;
	}
	//设置当前网段的广播地址 
    broadcast_addr.sin_family = AF_INET;
    broadcast_addr.sin_port = htons(broadcast_port);
    broadcast_addr.sin_addr.s_addr = inet_addr("10.0.0.255");  //设置为广播地址
	while(run_flag > 0)
	{
		std::cout << "heart loop!" << std::endl;
		sendto(broadcast_socket_fd, buf, strlen(buf), 0, (struct sockaddr *)&broadcast_addr, sizeof(broadcast_addr)); 
		sleep(1);
	}
	strcpy(buf, "LPR Stop!");
	sendto(broadcast_socket_fd, buf, strlen(buf), 0, (struct sockaddr *)&broadcast_addr, sizeof(broadcast_addr)); 
	close(broadcast_socket_fd);
	std::cout << "upd broadcast thread quit" << std::endl;
	return NULL;
}

static void * upd_recv_msg(void *arg)
{
	int ret = 0;
	int *socket_fd = (int *)arg;//通信的socket
	struct sockaddr_in  src_addr = {0};  //用来存放对方(信息的发送方)的IP地址信息
	int len = sizeof(src_addr);	//地址信息的大小
	char msg[1024] = {0};//消息缓冲区
	while(run_flag > 0)
	{
		ret = recvfrom(*socket_fd, msg, sizeof(msg), 0, (struct sockaddr *)&src_addr, (socklen_t*)len);
		if(ret > 0)
		{
			printf("[%s:%d]",inet_ntoa(src_addr.sin_addr),ntohs(src_addr.sin_port));
			printf("msg=%s\n",msg);
			if(strcmp(msg, "exit") == 0 || strcmp(msg, "") == 0)
			{
				run_flag = 0;
				break;
			}
			memset(msg, 0, sizeof(msg));//清空存留消息	
		}
	}
	//关闭通信socket
	close(*socket_fd);
	std::cout << "upd recv msg thread quit" << std::endl;
	return NULL;
}

static int start_all_lpr(global_control_param_t *G_param)
{
	int rval = 0;
	//pthread_t heart_pthread_id = 0;
	pthread_t pc_recv_thread_id = 0;
	// pthread_t save_pthread_id = 0;
	pthread_t ssd_pthread_id = 0;
	pthread_t lpr_pthread_id = 0;
	ssd_lpr_thread_params_t lpr_thread_params;
	ssd_lpr_thread_params_t ssd_thread_params;

	ea_tensor_t *img_tensor = NULL;
	ea_img_resource_data_t data;

	int upd_port = 9999;
	int udp_socket_fd = 0;
	struct sockaddr_in  local_addr = {0};
	struct timeval timeout;

	if(tof_geter.start() < 0)
	{
		rval = -1;
		run_flag = 0;
	}
	std::cout << "start tof success" << std::endl;

	udp_socket_fd = socket(AF_INET,SOCK_DGRAM,0);
	if(udp_socket_fd < 0 )
	{
		perror("creat socket fail\n");
		rval = -1;
		run_flag = 0;
	}
    timeout.tv_sec = 0;//秒
    timeout.tv_usec = 100000;//微秒
    if (setsockopt(udp_socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == -1) {
        perror("setsockopt failed:");
		rval = -1;
		run_flag = 0;
    }
	bzero(&local_addr, sizeof(local_addr));
	local_addr.sin_family  = AF_INET;
	local_addr.sin_port	= htons(upd_port);
	local_addr.sin_addr.s_addr = INADDR_ANY;
	rval = bind(udp_socket_fd,(struct sockaddr*)&local_addr,sizeof(local_addr));
	if(rval < 0)
	{
		perror("bind fail!");
		close(udp_socket_fd);
		rval = -1;
		run_flag = 0;
	}

	std::cout << "start_ssd_lpr" << std::endl;
	do {
		pthread_mutex_init(&result_mutex, NULL);
		pthread_mutex_init(&ssd_mutex, NULL);

		pthread_mutex_init(&image_buffer.lock, NULL);  
		pthread_cond_init(&image_buffer.notempty, NULL);  
		pthread_cond_init(&image_buffer.notfull, NULL);  
		image_buffer.readpos = 0;  
		image_buffer.writepos = 0;

		memset(&lpr_thread_params, 0 , sizeof(lpr_thread_params));
		memset(&data, 0, sizeof(data));
		RVAL_OK(ea_img_resource_hold_data(G_param->img_resource, &data));
		RVAL_ASSERT(data.tensor_group != NULL);
		RVAL_ASSERT(data.tensor_num >= 1);
		img_tensor = data.tensor_group[G_param->lpr_pyd_idx];
		lpr_thread_params.height = ea_tensor_shape(img_tensor)[2];
		lpr_thread_params.width = ea_tensor_shape(img_tensor)[3];
		lpr_thread_params.pitch = ea_tensor_pitch(img_tensor);
		lpr_thread_params.G_param = G_param;
		img_tensor = data.tensor_group[G_param->ssd_pyd_idx];
		ssd_thread_params.height = ea_tensor_shape(img_tensor)[2];
		ssd_thread_params.width = ea_tensor_shape(img_tensor)[3];
		ssd_thread_params.pitch = ea_tensor_pitch(img_tensor);
		ssd_thread_params.G_param = G_param;
		RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &data));
		rval = pthread_create(&ssd_pthread_id, NULL, run_ssd_pthread, (void*)&ssd_thread_params);
		RVAL_ASSERT(rval == 0);
		rval = pthread_create(&lpr_pthread_id, NULL, run_lpr_pthread, (void*)&lpr_thread_params);
		RVAL_ASSERT(rval == 0);
		// rval = pthread_create(&save_pthread_id, NULL, save_video_pthread, NULL);
		// RVAL_ASSERT(rval == 0);
		rval = pthread_create(&pc_recv_thread_id, NULL, upd_recv_msg, (void*)&udp_socket_fd);
		RVAL_ASSERT(rval == 0);
		// rval = pthread_create(&heart_pthread_id, NULL, upd_broadcast_send, NULL);
		// RVAL_ASSERT(rval == 0);
	} while (0);
	std::cout << "start_ssd_lpr success" << std::endl;

	point_cloud_process(G_param, &udp_socket_fd);
	//offline_point_cloud_process();

	if (lpr_pthread_id > 0) {
		pthread_join(lpr_pthread_id, NULL);
	}
	if (ssd_pthread_id > 0) {
		pthread_join(ssd_pthread_id, NULL);
	}
	// if (save_pthread_id > 0) {
	// 	pthread_join(save_pthread_id, NULL);
	// }
	if (pc_recv_thread_id > 0) {
		pthread_join(pc_recv_thread_id, NULL);
	}
	// if (heart_pthread_id > 0) {
	// 	pthread_join(heart_pthread_id, NULL);
	// }
	pthread_mutex_destroy(&result_mutex);
	pthread_mutex_destroy(&ssd_mutex);
	pthread_mutex_destroy(&image_buffer.lock);
    pthread_cond_destroy(&image_buffer.notempty);
    pthread_cond_destroy(&image_buffer.notfull);
	std::cout << "Main thread quit" << std::endl;
	return rval;
}

static void sigstop(int signal_number)
{
	run_lpr = 0;
	run_flag = 0;
	tof_geter.stop();
	// pthread_cond_signal(&image_buffer.notfull);
	// pthread_cond_signal(&image_buffer.notempty);
	printf("sigstop msg, exit live mode\n");
	return;
}

static int env_init(global_control_param_t *G_param)
{
	int rval = 0;
	std::cout << "env_init" << std::endl;
	do {
		RVAL_OK(ea_env_open(EA_ENV_ENABLE_IAV
			| EA_ENV_ENABLE_CAVALRY
			| EA_ENV_ENABLE_VPROC
			| EA_ENV_ENABLE_NNCTRL
			| EA_ENV_ENABLE_OSD_STREAM));
		G_param->img_resource = ea_img_resource_new(EA_PYRAMID,
			(void *)(unsigned long)G_param->channel_id);
		RVAL_ASSERT(G_param->img_resource != NULL);
		RVAL_OK(init_overlay_tool(G_param->stream_id,
			G_param->overlay_x_offset, G_param->overlay_highlight_sec,
			G_param->overlay_clear_sec, G_param->overlay_text_width_ratio,
			G_param->draw_plate_num, G_param->debug_en));
		RVAL_OK(init_state_buffer_param(&G_param->ssd_result_buf,
			G_param->state_buf_num, (uint16_t)sizeof(ea_img_resource_data_t),
			MAX_DETECTED_LICENSE_NUM, (G_param->debug_en >= INFO_LEVEL)));
		pthread_mutex_init(&G_param->access_buffer_mutex, NULL);
		sem_init(&G_param->sem_readable_buf, 0, 0);
		RVAL_OK(ea_set_preprocess_priority_on_current_process(VPROC_PRIORITY, EA_VP));
	} while(0);
	std::cout << "env_init success" << std::endl;
	return rval;
}

static void env_deinit(global_control_param_t *G_param)
{
	pthread_mutex_destroy(&G_param->access_buffer_mutex);
	sem_destroy(&G_param->sem_readable_buf);
	deinit_state_buffer_param(&G_param->ssd_result_buf);
	deinit_overlay_tool();
	ea_img_resource_free(G_param->img_resource);
	G_param->img_resource = NULL;
	ea_env_close();
}

int main(int argc, char **argv)
{
	int rval = 0;
	global_control_param_t G_param;

	signal(SIGINT, sigstop);
	signal(SIGQUIT, sigstop);
	signal(SIGTERM, sigstop);
	if(tof_geter.open_tof() == 0)
	{
		do {
			RVAL_OK(init_param(&G_param));
			RVAL_OK(env_init(&G_param));
			RVAL_OK(start_all_lpr(&G_param));
		}while(0);
		env_deinit(&G_param);
	}
	printf("All Quit.\n");
	return rval;
}


