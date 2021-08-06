/*******************************************************************************
 * test_ssd_lpr.c
 *
 * History:
 *    2020/03/31  - [Junshuai ZHU] created
 *
 * Copyright (c) 2020 Ambarella International LP
 *
 * This file and its contents ( "Software" ) are protected by intellectual
 * property rights including, without limitation, U.S. and/or foreign
 * copyrights. This Software is also the confidential and proprietary
 * information of Ambarella International LP and its licensors. You may not use, reproduce,
 * disclose, distribute, modify, or otherwise prepare derivative works of this
 * Software or any portion thereof except pursuant to a signed license agreement
 * or nondisclosure agreement with Ambarella International LP or its authorized affiliates.
 * In the absence of such an agreement, you agree to promptly notify and return
 * this Software to Ambarella International LP.
 *
 * This file includes sample code and is only for internal testing and evaluation.  If you
 * distribute this sample code (whether in source, object, or binary code form), it will be
 * without any warranty or indemnity protection from Ambarella International LP or its affiliates.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF NON-INFRINGEMENT,
 * MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL AMBARELLA INTERNATIONAL LP OR ITS AFFILIATES BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; COMPUTER FAILURE OR MALFUNCTION; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
******************************************************************************/

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
//#include <iav_ioctl.h>
#include <eazyai.h>
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
volatile int run_ssd = 0;

TOFAcquisition tof_geter;

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
//			if (G_param->debug_en >= INFO_LEVEL) {
			if (1) {
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
			while(run_ssd > 0)
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
			RVAL_OK(ea_img_resource_hold_data(G_param->img_resource, &data));
			RVAL_ASSERT(data.tensor_group != NULL);
			RVAL_ASSERT(data.tensor_num >= 1);
			img_tensor = data.tensor_group[G_param->ssd_pyd_idx];
			dsp_pts = data.dsp_pts;
			// SAVE_TENSOR_IN_DEBUG_MODE("SSD_pyd.jpg", img_tensor, debug_en);
			if(frame_number % 40 == 0)
			{
				SAVE_TENSOR_GROUP_IN_DEBUG_MODE("image", frame_number, img_tensor, 3);
			}
			frame_number++;

			start_time = gettimeus();

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
			}
			RVAL_OK(set_overlay_bbox(&bbox_list));
			RVAL_OK(show_overlay(dsp_pts));
			TIME_MEASURE_END("[SSD] post-process time", debug_en);

			sum_time += (gettimeus() - start_time);
			++loop_count;
			if (loop_count == TIME_MEASURE_LOOPS) {
				printf("SSD average time [per %d loops]: %f ms\n",
					TIME_MEASURE_LOOPS, sum_time / (1000 * TIME_MEASURE_LOOPS));
				sum_time = 0;
				loop_count = 1;
			}
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

	return NULL;
}

static int dump_ply(const char* save_path, const TOFAcquisition::PointCloud &src_cloud)
{
	char ply_header[100];
	sprintf(ply_header, "element vertex %ld\n", src_cloud.size());
	FILE *fptr;
	fptr = fopen(save_path, "w");

	fprintf(fptr, "ply\n");
	fprintf(fptr, "format ascii 1.0\n");
	fprintf(fptr, "%s", ply_header);
	fprintf(fptr, "property double x\nproperty double y\nproperty double z\n");
	fprintf(fptr, "property uchar red\nproperty uchar green\nproperty uchar blue\n");
	fprintf(fptr, "end_header\n");
	for (size_t i = 0; i < src_cloud.size(); i++)
	{
		fprintf(fptr, "%f %f %f ", src_cloud[i].x, src_cloud[i].y, src_cloud[i].z);
		fprintf(fptr, "%d %d %d\n", 255, 0, 0);
	}
	fclose(fptr);
	std::cout << "save ply OK..." << std::endl;
	return 0;
}

static void point_cloud_process()
{
	unsigned long long int frame_number = 0;
	TOFAcquisition::PointCloud src_cloud;
	while(run_flag > 0)
	{
		tof_geter.get_tof_data(src_cloud);
		frame_number++;
		if(src_cloud.size() > 0)
		{
			run_ssd = 1;
			if(frame_number % 10 == 0)
			{
				std::stringstream filename;
				filename << "point_cloud" << frame_number << ".ply";
				dump_ply(filename.str().c_str(), src_cloud);
			}
		}
		else
		{
			run_ssd = 0;
		}
		std::cout << "Point Cloud:" << frame_number << std::endl;
	}
	std::cout << "stop point cloud process" << std::endl;
}

static int start_all_lpr(global_control_param_t *G_param)
{
	int rval = 0;
	pthread_t ssd_pthread_id = 0;
	pthread_t lpr_pthread_id = 0;
	ssd_lpr_thread_params_t lpr_thread_params;
	ssd_lpr_thread_params_t ssd_thread_params;

	ea_tensor_t *img_tensor = NULL;
	ea_img_resource_data_t data;

	std::cout << "start_ssd_lpr" << std::endl;
	do {
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
	} while (0);
	std::cout << "start_ssd_lpr success" << std::endl;

	if(tof_geter.start() < 0)
	{
		rval = -1;
		run_flag = 0;
	}
	std::cout << "start tof success" << std::endl;
	point_cloud_process();

	if (lpr_pthread_id > 0) {
		pthread_join(lpr_pthread_id, NULL);
	}
	if (ssd_pthread_id > 0) {
		pthread_join(ssd_pthread_id, NULL);
	}
	std::cout << "Main thread quit" << std::endl;
	return rval;
}

static void sigstop(int signal_number)
{
	run_flag = 0;
	tof_geter.stop();
	printf("sigstop msg, exit live mode\n");
	return;
}

static int env_init(global_control_param_t *G_param)
{
	int rval = 0;;
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

	return;
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


