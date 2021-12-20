#ifndef _DET_LPR_PROCESS_H_
#define _DET_LPR_PROCESS_H_

#include "cnn_lpr/common/common_process.h"
#include "cnn_lpr/lpr/ssd.h"
#include "cnn_lpr/det2d/yolov5.h"

typedef struct lpr_thread_params_s {
	uint16_t width;
	uint16_t height;
	uint16_t pitch;
	uint16_t reserved;
	global_control_param_t *G_param;
}lpr_thread_params_t;

void upscale_normalized_rectangle(float x_min, float y_min,
	float x_max, float y_max, float w_ratio, float h_ratio,
	bbox_param_t *bbox_scaled);

int init_ssd(SSD_ctx_t *SSD_ctx, global_control_param_t *G_param,
	uint32_t buffer_h, uint32_t buffer_w);

int init_yolov5(yolov5_t *live_ctx, global_control_param_t *G_param);

float overlap(float x1, float w1, float x2, float w2);

float cal_iou(std::vector<float> box, std::vector<float> truth);

#endif // _DET_LPR_PROCESS_H_