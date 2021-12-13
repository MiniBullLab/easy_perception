#ifndef _COMMON_PROCESS_H_
#define _COMMON_PROCESS_H_

#include <glog/logging.h>
#include <glog/raw_logging.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>

#include <lib_data_process.h>
#include <iav_ioctl.h>
#include <eazyai.h>
#include <tensor_private.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "cnn_lpr/lpr/utils.hpp"
#include "cnn_lpr/lpr/ssd_lpr_common.h"
#include "cnn_lpr/lpr/state_buffer.h"
#include "cnn_lpr/show/overlay_tool.h"


EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);

enum cavalry_priotiry {
	SSD_PRIORITY,
	VPROC_PRIORITY,
	LPR_PRIORITY,
	PRIORITY_NUM
};

typedef struct global_control_param_s {
	// cmd line param
	uint8_t channel_id;
	uint8_t stream_id;
	uint32_t state_buf_num;
	float overlay_text_width_ratio;
	float overlay_x_offset;
	uint16_t overlay_highlight_sec;
	uint16_t overlay_clear_sec;
	uint32_t verbose; /* network print time */
	uint32_t debug_en; /* 0: disable; 1: time measure,2: log; 3: run once & save picture */
	uint32_t abort_if_preempted;
	ea_img_resource_t *img_resource;

	// run time control
	state_buffer_param_t ssd_result_buf;
	pthread_mutex_t access_buffer_mutex;
	sem_t sem_readable_buf;
	pthread_mutex_t vp_access_lock;
} global_control_param_t;

int init_param(global_control_param_t *G_param);

int env_init(global_control_param_t *G_param);

void env_deinit(global_control_param_t *G_param);

int tensor2mat_yuv2bgr_nv12(ea_tensor_t *tensor, cv::Mat &bgr);

void swapYUV_I420toNV12(const unsigned char* i420bytes, unsigned char* nv12bytes, int width, int height);

int mat2tensor_yuv_nv12(cv::Mat &yuv_i420, ea_tensor_t *tensor);

int create_yuv_nv12_tensor(const unsigned char* addr, const int width, const int height, ea_tensor_t *tensor);

void fill_data(unsigned char* addr, int data);

#endif // _COMMON_PROCESS_H_