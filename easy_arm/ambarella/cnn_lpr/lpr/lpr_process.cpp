#include "lpr_process.h"
#include <iostream>

const static std::string lpr_model_path = "/data/lpr/segfree_inception_cavalry.bin";
const static std::vector<std::string> lpr_input_name = {"data"};
const static std::vector<std::string> lpr_output_name = {"prob"};

const static std::string lphm_model_path = "/data/lpr/LPHM_cavalry.bin";
const static std::vector<std::string> lphm_input_name = {"data"};
const static std::vector<std::string> lphm_output_name = {"dense"};

int ssd_critical_resource(
	dproc_ssd_detection_output_result_t *amba_ssd_result,
	ea_img_resource_data_t* imgs_data_addr, int ssd_result_num,
	state_buffer_t *ssd_mid_buf, global_control_param_t *G_param)
{
	int i, rval = 0;
	ea_img_resource_data_t covered_imgs_addr;
	uint8_t buffer_covered = 0;
	float score  = 0;

	do {
		ssd_result_num = min(MAX_DETECTED_LICENSE_NUM, ssd_result_num);
		for (i = 0; i < ssd_result_num; i++) {
			if(amba_ssd_result[i].score > score)
			{
				ssd_mid_buf->bbox_param[0].norm_max_x =
				amba_ssd_result[i].bbox.x_max;
				ssd_mid_buf->bbox_param[0].norm_max_y =
					amba_ssd_result[i].bbox.y_max;
				ssd_mid_buf->bbox_param[0].norm_min_x =
					amba_ssd_result[i].bbox.x_min;
				ssd_mid_buf->bbox_param[0].norm_min_y =
					amba_ssd_result[i].bbox.y_min;
				ssd_mid_buf->bbox_param[0].p1_x = 0;
				ssd_mid_buf->bbox_param[0].p1_y = 0;
				ssd_mid_buf->bbox_param[0].p2_x = 0;
				ssd_mid_buf->bbox_param[0].p2_y = 0;
				ssd_mid_buf->bbox_param[0].p3_x = 0;
				ssd_mid_buf->bbox_param[0].p3_y = 0;
				ssd_mid_buf->bbox_param[0].p4_x = 0;
				ssd_mid_buf->bbox_param[0].p4_y = 0;
				score = amba_ssd_result[i].score;
				LOG(INFO) << "best bbox: " << amba_ssd_result[i].bbox.x_min << " " << amba_ssd_result[i].bbox.y_min << " " \
				 << amba_ssd_result[i].bbox.x_max << " " << amba_ssd_result[i].bbox.y_max;
				LOG(WARNING) << "best bbox score: " << score;
			}
		}
		if(score > 0)
		{
			ssd_mid_buf->object_num = 1;
		}
		else
		{
			ssd_mid_buf->object_num = 0;
		}
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

int yolov5_critical_resource(
	landmark_yolov5_det_t *yolov5_result,
	ea_img_resource_data_t* imgs_data_addr, int result_num,
	state_buffer_t *ssd_mid_buf, global_control_param_t *G_param)
{
	int i, rval = 0;
	ea_img_resource_data_t covered_imgs_addr;
	uint8_t buffer_covered = 0;
	float score  = 0;

	do {
		result_num = min(MAX_DETECTED_LICENSE_NUM, result_num);
		for (i = 0; i < result_num; i++) {
			if(yolov5_result[i].score > score)
			{
				ssd_mid_buf->bbox_param[0].norm_max_x = yolov5_result[i].x_end;
				ssd_mid_buf->bbox_param[0].norm_max_y = yolov5_result[i].y_end;
				ssd_mid_buf->bbox_param[0].norm_min_x = yolov5_result[i].x_start;
				ssd_mid_buf->bbox_param[0].norm_min_y = yolov5_result[i].y_start;
				if(yolov5_result[i].p1_x > yolov5_result[i].p2_x && \
						yolov5_result[i].p4_x > yolov5_result[i].p3_x)
				{
					ssd_mid_buf->bbox_param[0].p1_x = yolov5_result[i].p1_x;
					ssd_mid_buf->bbox_param[0].p1_y = yolov5_result[i].p1_y;
					ssd_mid_buf->bbox_param[0].p2_x = yolov5_result[i].p2_x;
					ssd_mid_buf->bbox_param[0].p2_y = yolov5_result[i].p2_y;
					ssd_mid_buf->bbox_param[0].p3_x = yolov5_result[i].p3_x;
					ssd_mid_buf->bbox_param[0].p3_y = yolov5_result[i].p3_y;
					ssd_mid_buf->bbox_param[0].p4_x = yolov5_result[i].p4_x;
					ssd_mid_buf->bbox_param[0].p4_y = yolov5_result[i].p4_y;
				}
				else
				{
					ssd_mid_buf->bbox_param[0].p1_x = yolov5_result[i].p2_x;
					ssd_mid_buf->bbox_param[0].p1_y = yolov5_result[i].p2_y;
					ssd_mid_buf->bbox_param[0].p2_x = yolov5_result[i].p1_x;
					ssd_mid_buf->bbox_param[0].p2_y = yolov5_result[i].p1_y;
					ssd_mid_buf->bbox_param[0].p3_x = yolov5_result[i].p4_x;
					ssd_mid_buf->bbox_param[0].p3_y = yolov5_result[i].p4_y;
					ssd_mid_buf->bbox_param[0].p4_x = yolov5_result[i].p3_x;
					ssd_mid_buf->bbox_param[0].p4_y = yolov5_result[i].p3_y;
				}

				score = yolov5_result[i].score;
				LOG(INFO) << "best bbox: " << yolov5_result[i].x_start << " " << yolov5_result[i].y_start << " " \
				 << yolov5_result[i].x_end << " " << yolov5_result[i].y_end;
				LOG(WARNING) << "best bbox score: " << score;
			}
		}
		if(score > 0)
		{
			ssd_mid_buf->object_num = 1;
		}
		else
		{
			ssd_mid_buf->object_num = 0;
		}
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

int lpr_critical_resource(uint16_t *license_num, bbox_param_t *bbox_param,
	state_buffer_t *ssd_mid_buf, global_control_param_t *G_param)
{
	int i, rval = 0;

	do {
		RVAL_OK(read_state_buffer(ssd_mid_buf, &G_param->ssd_result_buf,
			&G_param->access_buffer_mutex, &G_param->sem_readable_buf));
		*license_num = ssd_mid_buf->object_num;
		if (*license_num > MAX_DETECTED_LICENSE_NUM) {
			LOG(ERROR) << "license_num " << *license_num << " > " << MAX_DETECTED_LICENSE_NUM;
			rval = -1;
			break;
		}
		for (i = 0; i < *license_num; ++i) {
			bbox_param[i].norm_min_x = ssd_mid_buf->bbox_param[i].norm_min_x;
			bbox_param[i].norm_min_y = ssd_mid_buf->bbox_param[i].norm_min_y;
			bbox_param[i].norm_max_x = ssd_mid_buf->bbox_param[i].norm_max_x;
			bbox_param[i].norm_max_y = ssd_mid_buf->bbox_param[i].norm_max_y;
			bbox_param[i].p1_x = ssd_mid_buf->bbox_param[i].p1_x;
			bbox_param[i].p1_y = ssd_mid_buf->bbox_param[i].p1_y;
			bbox_param[i].p2_x = ssd_mid_buf->bbox_param[i].p2_x;
			bbox_param[i].p2_y = ssd_mid_buf->bbox_param[i].p2_y;
			bbox_param[i].p3_x = ssd_mid_buf->bbox_param[i].p3_x;
			bbox_param[i].p3_y = ssd_mid_buf->bbox_param[i].p3_y;
			bbox_param[i].p4_x = ssd_mid_buf->bbox_param[i].p4_x;
			bbox_param[i].p4_y = ssd_mid_buf->bbox_param[i].p4_y;
		}
		if (G_param->debug_en >= INFO_LEVEL) {
			LOG(INFO) << "------------------------------";
			LOG(INFO) << "LPR got bboxes:";
			for (i = 0; i < *license_num; ++i) {
				LOG(INFO) << i << " " << bbox_param[i].norm_min_x << "," << bbox_param[i].norm_min_y << "|" << \
				bbox_param[i].norm_max_x << "," << bbox_param[i].norm_max_y;
			}
			LOG(INFO) << "------------------------------";
		}
	} while (0);

	return rval;
}

int init_LPR(LPR_ctx_t *LPR_ctx, global_control_param_t *G_param)
{
	int rval = 0;
	LOG(INFO) << "init_LPR";
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
	LOG(INFO) << "init_LPR success";
	return rval;
}

void bbox_list_process(const bbox_list_t *list_lpr_bbox, bbox_list_t *result_bbox)
{
	std::vector<bbox_param_t> result;
	int input_count = list_lpr_bbox->bbox_num;
	int output_count = 0;
	ResultRect input_bbox[MAX_OVERLAY_PLATE_NUM] = {0};
	ResultRect output_bbox[MAX_OVERLAY_PLATE_NUM] = {0};
	result_bbox->bbox_num = 0;
	if(list_lpr_bbox->bbox_num <= 0)
	{
		return;
	}
	for(size_t i = 0; i < list_lpr_bbox->bbox_num; i++)
	{
		input_bbox[i].x = (long)list_lpr_bbox->bbox[i].norm_min_x;
		input_bbox[i].y = (long)list_lpr_bbox->bbox[i].norm_min_y;
		input_bbox[i].width = (long)(list_lpr_bbox->bbox[i].norm_max_x - list_lpr_bbox->bbox[i].norm_min_x);
		input_bbox[i].height = (long)(list_lpr_bbox->bbox[i].norm_max_y - list_lpr_bbox->bbox[i].norm_min_y);
		input_bbox[i].confidence = 0;
	}
	if(input_count > 0)
	{
		clusteringRect(input_bbox, input_count, 0.12f, output_bbox, &output_count);
		for(size_t i = 0; i < output_count; i++)
		{
			bbox_param_t bbox = {0};
			bbox.norm_min_x = (float)output_bbox[i].x;
			bbox.norm_min_y = (float)output_bbox[i].y;
			bbox.norm_max_x = (float)(output_bbox[i].x + output_bbox[i].width);
			bbox.norm_max_y = (float)(output_bbox[i].y + output_bbox[i].height);
			result_bbox->bbox[i] = bbox;
		}
		result_bbox->bbox_num = output_count;
	}
}

void draw_overlay_preprocess(draw_plate_list_t *draw_plate_list,
	license_list_t *license_result, bbox_param_t *bbox_param, uint32_t debug_en)
{
	uint32_t i = 0;
	int draw_num = 0;
	bbox_param_t scaled_bbox_draw[MAX_DETECTED_LICENSE_NUM];
	license_plate_t *plates = draw_plate_list->license_plate;
	license_info_t *license_info = license_result->license_info;

	license_result->license_num = min(license_result->license_num, MAX_OVERLAY_PLATE_NUM);

	
	for (i = 0; i < license_result->license_num; ++i) {
		size_t char_len = strlen(license_info[i].text);
		if (license_info[i].conf > DEFAULT_LPR_CONF_THRES && \
			(char_len == 9 || char_len == 10)) {
			upscale_normalized_rectangle(bbox_param[i].norm_min_x, bbox_param[i].norm_min_y,
			bbox_param[i].norm_max_x, bbox_param[i].norm_max_y,
				DRAW_LICNESE_UPSCALE_W, DRAW_LICNESE_UPSCALE_H, &scaled_bbox_draw[i]);
			plates[draw_num].bbox.norm_min_x = scaled_bbox_draw[i].norm_min_x;
			plates[draw_num].bbox.norm_min_y = scaled_bbox_draw[i].norm_min_y;
			plates[draw_num].bbox.norm_max_x = scaled_bbox_draw[i].norm_max_x;
			plates[draw_num].bbox.norm_max_y = scaled_bbox_draw[i].norm_max_y;
			plates[draw_num].conf = license_info[i].conf;
			memset(plates[draw_num].text, 0, sizeof(plates[draw_num].text));
			snprintf(plates[draw_num].text, sizeof(plates[draw_num].text), "%s", license_info[i].text);
			plates[draw_num].text[sizeof(plates[draw_num].text) - 1] = '\0';
			++draw_num;
			if (debug_en > 0) {
				LOG(INFO) << "Drawed license:" << license_info[i].text << "," << license_info[i].conf;
			}
		}
	}
	draw_plate_list->license_num = draw_num;

	return;
}