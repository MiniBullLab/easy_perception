#ifndef _LPR_PROCESS_H_
#define _LPR_PROCESS_H_

#include "cnn_lpr/common/common_process.h"
#include "cnn_lpr/lpr/lpr.hpp"
#include "cnn_lpr/lpr/det_lpr_process.h"

#include "cnn_lpr/clustering/clustering_rect.h"

#if defined(OLD_CODE) || defined(USE_OLD_REC)
#define DEFAULT_LPR_CONF_THRES		(0.9f)
#else
#define DEFAULT_LPR_CONF_THRES		(0.5f)
#endif
#define CHINESE_LICENSE_STR_LEN		(9)
#define DRAW_LICNESE_UPSCALE_H		(1.0f)
#define DRAW_LICNESE_UPSCALE_W		(0.2f)

int ssd_critical_resource(
	dproc_ssd_detection_output_result_t *amba_ssd_result,
	ea_img_resource_data_t* imgs_data_addr, int ssd_result_num,
	state_buffer_t *ssd_mid_buf, global_control_param_t *G_param);

int yolov5_critical_resource(
	landmark_yolov5_det_t *yolov5_result,
	ea_img_resource_data_t* imgs_data_addr, int result_num,
	state_buffer_t *ssd_mid_buf, global_control_param_t *G_param);

int lpr_critical_resource(uint16_t *license_num, bbox_param_t *bbox_param,
	state_buffer_t *ssd_mid_buf, global_control_param_t *G_param);

int init_LPR(LPR_ctx_t *LPR_ctx, global_control_param_t *G_param);

void bbox_list_process(const bbox_list_t *list_lpr_bbox, bbox_list_t *result_bbox);

void draw_overlay_preprocess(draw_plate_list_t *draw_plate_list,
	license_list_t *license_result, bbox_param_t *bbox_param, uint32_t debug_en);

#endif // _LPR_PROCESS_H_