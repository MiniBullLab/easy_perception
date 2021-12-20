/*******************************************************************************
 * yolov5.c
 *
 * History:
 *  2020/09/28  - [Du You] created
 *
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

#include <config.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <signal.h>
#include <fcntl.h>

#include <eazyai.h>

#include "yolov5.h"

#define SIGMOID(x)	(1.0 / (1.0 + exp(-(x))))

EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);

enum {
	NMS_INIT,
	NMS_CHOSEN,
	NMS_DISCARD,
};

static float const yolov5_anchors[YOLOV5_FEATURE_MAP_NUM][YOLOV5_ANCHOR_NUM][2] = {
	{
		{116, 90}, {156, 198}, {373, 326}	// anchor box of feature map 1
	},
	{
		{30, 61}, {62, 45}, {59, 119}		// anchor box of feature map 2
	},
	{
		{10, 13}, {16, 30}, {33, 23}		// anchor box of feature map 3
	}
};

static float const landmark_yolov5_anchors[YOLOV5_FEATURE_MAP_NUM][YOLOV5_ANCHOR_NUM][2] = {
	{
		{146, 217}, {231, 300}, {335, 433}	// anchor box of feature map 1
	},
	{
		{23, 29}, {43, 55}, {73, 105}		// anchor box of feature map 2
	},
	{
		{4, 5}, {8, 10}, {13, 16}		// anchor box of feature map 3
	}
};

static int landmark_nms(float *x1y1x2y2score, void *aux, size_t aux_element_size, int num, \
                        float threshold, int use_iou_min, int top_k, \
	                    float *out_x1y1x2y2score, void *out_aux, int *out_num)
{
	int rval = EA_SUCCESS;
	float *area = NULL;
	int *sort = NULL;
	int *status = NULL;
	int i, k;
	int temp;
	int high_ind;
	float area_high;
	float area_i;
	float max_x;
	float max_y;
	float min_x;
	float min_y;
	float iou_width;
	float iou_height;
	float iou_area;
	float iou_ratio;
	int chosen_count = 0;

	do {
		RVAL_ASSERT(x1y1x2y2score != NULL);
		RVAL_ASSERT(out_x1y1x2y2score != NULL);
		RVAL_ASSERT(out_num != NULL);

		*out_num = 0;

		if (num <= 0) {
			break;
		}

		area = (float *)malloc(sizeof(float) * num);
		RVAL_ASSERT(area != NULL);

		for (i = 0; i < num; i++) {
			area[i] = (x1y1x2y2score[i * 13 + 2] - x1y1x2y2score[i * 13 + 0]) *
				(x1y1x2y2score[i * 13 + 3] - x1y1x2y2score[i * 13 + 1]);
			area[i] = max(0, area[i]);
		}

		sort = (int *)malloc(sizeof(int) * num);
		RVAL_ASSERT(sort != NULL);
		for (i = 0; i < num; i++) {
			sort[i] = i;
		}

		for (i = 0; i < num - 1; i++) {
			for (k = i + 1; k < num; k++) {
				if (x1y1x2y2score[sort[i] * 13 + 4] < x1y1x2y2score[sort[k] * 13 + 4]) {
					temp = sort[i];
					sort[i] = sort[k];
					sort[k] = temp;
				}
			}
		}

		status = (int *)malloc(sizeof(int) * num);
		RVAL_ASSERT(status != NULL);
		for (i = 0; i < num; i++) {
			status[i] = NMS_INIT;
		}

		while (1) {
			high_ind = -1;
			for (i = 0; i < num; i++) {
				if (status[sort[i]] == NMS_INIT) {
					high_ind = sort[i];
					break;
				}
			}

			if (high_ind == -1) {
				break;
			}

			status[high_ind] = NMS_CHOSEN;
			chosen_count++;
			if (top_k > 0 && chosen_count >= top_k) {
				break;
			}

			for (i = 0; i < num; i++) {
				if (status[i] == NMS_INIT) {
					area_high = area[high_ind];
					area_i = area[i];
					max_x = max(x1y1x2y2score[i * 13 + 0] , x1y1x2y2score[high_ind * 13 + 0]);
					max_y = max(x1y1x2y2score[i * 13 + 1] , x1y1x2y2score[high_ind * 13 + 1]);
					min_x = min(x1y1x2y2score[i * 13 + 2] , x1y1x2y2score[high_ind * 13 + 2]);
					min_y = min(x1y1x2y2score[i * 13 + 3] , x1y1x2y2score[high_ind * 13 + 3]);

					iou_width = ((min_x - max_x) > 0) ? (min_x - max_x) : 0;
					iou_height = ((min_y - max_y) > 0) ? (min_y - max_y) : 0;
					iou_area = iou_width * iou_height;

					if (use_iou_min) {
						iou_ratio = iou_area / min(area_high, area_i);
					}
					else { // use iou union
						iou_ratio = iou_area / (area_high + area_i - iou_area);
					}

					if(iou_ratio > threshold){
						status[i] = NMS_DISCARD;
					}
				}
			}
		}

		for (i = 0; i < num; i++) {
			if (status[sort[i]] == NMS_CHOSEN) {
				memcpy(&out_x1y1x2y2score[(*out_num) * 13], &x1y1x2y2score[sort[i] * 13], sizeof(float) * 13);
				if (aux) {
					memcpy(&((uint8_t *)out_aux)[(*out_num) * aux_element_size], &((uint8_t *)aux)[sort[i] * aux_element_size], aux_element_size);
				}
				*out_num += 1;
			}
		}
	} while (0);

	if (area) {
		free(area);
		area = NULL;
	}

	if (sort) {
		free(sort);
		sort = NULL;
	}

	if (status) {
		free(status);
		status = NULL;
	}

	return rval;
}

int yolov5_init(yolov5_t *yolov5, const yolov5_params_t *params)
{
	int rval = 0;
	// FILE *fp_label = NULL;
	// char *endl = NULL;
	int i;

	do {
		RVAL_OK(yolov5 != NULL);
		RVAL_ASSERT(params != NULL);
		RVAL_ASSERT(params->model_path != NULL);
		EA_LOG_SET_LOCAL(params->log_level);

		memset(yolov5, 0, sizeof(yolov5_t));
		yolov5->top_k = params->keep_top_k;
		yolov5->nms_threshold = params->nms_threshold;
		yolov5->conf_threshold = params->conf_threshold;
		yolov5->use_multi_cls = params->use_multi_cls;

		yolov5->net = ea_net_new(NULL);
		RVAL_ASSERT(yolov5->net != NULL);

		if (params->log_level == EA_LOG_LEVEL_VERBOSE) {
			ea_net_params(yolov5->net)->verbose_print = 1;
		}

		ea_net_config_input(yolov5->net, params->input_name);
		for (i = 0; i < YOLOV5_FEATURE_MAP_NUM; i++) {
			ea_net_config_output(yolov5->net, params->feature_map_names[i]);
		}
		RVAL_OK(ea_net_load(yolov5->net, EA_NET_LOAD_FILE, (void *)params->model_path, 1/*max_batch*/));
		yolov5->input_tensor = ea_net_input(yolov5->net, params->input_name);
		for (i = 0; i < YOLOV5_FEATURE_MAP_NUM; i++) {
			yolov5->feature_map_tensors[i] = ea_net_output(yolov5->net, params->feature_map_names[i]);
		}

		// // load label from file
		// fp_label = fopen(params->label_path, "r");
		// if (fp_label == NULL) {
		// 	EA_LOG_ERROR("can't open file %s\n", params->label_path);
		// 	rval = -1;
		// 	break;
		// }

		// yolov5->valid_label_count = 0;
		// for (i = 0; i < YOLOV5_MAX_LABEL_NUM; i++) {
		// 	if (fgets(yolov5->labels[i], YOLOV5_MAX_LABEL_LEN, fp_label) == NULL) {
		// 		break;
		// 	}

		// 	if (strlen(yolov5->labels[i]) >= YOLOV5_MAX_LABEL_LEN - 1) {
		// 		EA_LOG_ERROR("YOLOV5_MAX_LABEL_LEN %d is too small\n", YOLOV5_MAX_LABEL_LEN);
		// 		rval = -1;
		// 		break;
		// 	}

		// 	endl = strchr(yolov5->labels[i], '\n');
		// 	if (endl) {
		// 		endl[0] = '\0';
		// 	}

		// 	yolov5->valid_label_count++;
		// }

		RVAL_BREAK();

		// fclose(fp_label);
		// fp_label = NULL;

		// EA_LOG_NOTICE("label num: %d\n", yolov5->valid_label_count);
	} while (0);

	if (rval < 0) {
		// if (fp_label) {
		// 	fclose(fp_label);
		// 	fp_label = NULL;
		// }

		if (yolov5) {
			if (yolov5->net) {
				ea_net_free(yolov5->net);
				yolov5->net = NULL;
			}
		}
	}

	return rval;
}

void yolov5_deinit(yolov5_t *yolov5)
{
	if (yolov5) {
		if (yolov5->net) {
			ea_net_free(yolov5->net);
			yolov5->net = NULL;
		}
	}

	EA_LOG_NOTICE("yolov5_deinit\n");
}

ea_tensor_t *yolov5_input(yolov5_t *yolov5)
{
	return yolov5->input_tensor;
}

int yolov5_vp_forward(yolov5_t *yolov5)
{
	int rval = 0;

	do {
		RVAL_OK(ea_net_forward(yolov5->net, 1/*batch*/));
	} while (0);

	return rval;
}

int yolov5_arm_post_process(yolov5_t *yolov5, yolov5_result_t *result)
{
	int rval = 0;
	int class_num = ea_tensor_shape(yolov5->feature_map_tensors[0])[EA_C] / YOLOV5_ANCHOR_NUM - 5;
	int height[YOLOV5_FEATURE_MAP_NUM];
	int width[YOLOV5_FEATURE_MAP_NUM];
	int max_det_num_in_class = 0;
	uint8_t *feature_map_data;
	uint8_t *feature_map_data_in_anchor;
	uint8_t *x_data, *y_data, *w_data, *h_data, *box_conf_data, *cls_conf_data;
	uint8_t *x_data_w, *y_data_w, *w_data_w, *h_data_w, *box_conf_data_w, *cls_conf_data_w;
	const size_t *shape;
	size_t pitch;
	int stride_w, stride_h;
	float xywhscore[5];
	float cls_conf;
	float **x1y1x2y2score_in_class = NULL;
	int *valid_count_in_class = NULL;
	float **nms_x1y1x2y2score_in_class = NULL;
	int *nms_valid_count_in_class = NULL;
	int i, m, a, c, h, w;
	int best_cls;
	float best_cls_conf;

	do {
		for (m = 0; m < YOLOV5_FEATURE_MAP_NUM; m++) {
			RVAL_OK(ea_tensor_sync_cache(yolov5->feature_map_tensors[m], EA_VP, EA_CPU));
			height[m] = ea_tensor_shape(yolov5->feature_map_tensors[m])[EA_H];
			width[m] = ea_tensor_shape(yolov5->feature_map_tensors[m])[EA_W];
			max_det_num_in_class += height[m] * width[m] * YOLOV5_ANCHOR_NUM;
		}

		RVAL_BREAK();

		x1y1x2y2score_in_class = (float **)malloc(class_num * sizeof(float *));
		RVAL_ASSERT(x1y1x2y2score_in_class != NULL);
		for (c = 0; c < class_num; c++) {
			x1y1x2y2score_in_class[c] = (float *)malloc(max_det_num_in_class * 5 * sizeof(float));
			RVAL_ASSERT(x1y1x2y2score_in_class[c] != NULL);
		}

		RVAL_BREAK();

		valid_count_in_class = (int *)malloc(class_num * sizeof(int));
		RVAL_ASSERT(valid_count_in_class != NULL);
		memset(valid_count_in_class, 0, class_num * sizeof(int));

		for (m = 0; m < YOLOV5_FEATURE_MAP_NUM; m++) {
			feature_map_data = (uint8_t *)ea_tensor_data(yolov5->feature_map_tensors[m]);
			shape = ea_tensor_shape(yolov5->feature_map_tensors[m]);
			pitch = ea_tensor_pitch(yolov5->feature_map_tensors[m]);
			stride_w = ea_tensor_shape(yolov5->input_tensor)[EA_W] / shape[EA_W];
			stride_h = ea_tensor_shape(yolov5->input_tensor)[EA_H] / shape[EA_H];
			for (a = 0; a < YOLOV5_ANCHOR_NUM; a++) {
				feature_map_data_in_anchor = feature_map_data + shape[EA_H] * pitch * (class_num + 5) * a;
				x_data = feature_map_data_in_anchor;
				y_data = x_data + shape[EA_H] * pitch;
				w_data = y_data + shape[EA_H] * pitch;
				h_data = w_data + shape[EA_H] * pitch;
				box_conf_data = h_data + shape[EA_H] * pitch;
				if (yolov5->use_multi_cls) {
					for (c = 0; c < class_num; c++) {
						cls_conf_data = box_conf_data + shape[EA_H] * pitch + c * shape[EA_H] * pitch;
						for (h = 0; h < height[m]; h++) {
							x_data_w = x_data + h * pitch;
							y_data_w = y_data + h * pitch;
							w_data_w = w_data + h * pitch;
							h_data_w = h_data + h * pitch;
							x_data_w = x_data + h * pitch;
							box_conf_data_w = box_conf_data + h * pitch;
							cls_conf_data_w = cls_conf_data + h * pitch;
							for (w = 0; w < width[m]; w++) {
								xywhscore[4] = ((float *)box_conf_data_w)[w];
								xywhscore[4] = SIGMOID(xywhscore[4]);
								if (xywhscore[4] > yolov5->conf_threshold) {
									xywhscore[2] = ((float *)w_data_w)[w];
									xywhscore[2] = SIGMOID(xywhscore[2]);
									xywhscore[2] = pow(xywhscore[2] * 2.0, 2.0) * yolov5_anchors[m][a][0];
									if (xywhscore[2] > YOLOV5_MIN_WH && xywhscore[2] < YOLOV5_MAX_WH) {
										xywhscore[3] = ((float *)h_data_w)[w];
										xywhscore[3] = SIGMOID(xywhscore[3]);
										xywhscore[3] = pow(xywhscore[3] * 2.0, 2.0) * yolov5_anchors[m][a][1];
										if (xywhscore[3] > YOLOV5_MIN_WH && xywhscore[3] < YOLOV5_MAX_WH) {
											cls_conf = ((float *)cls_conf_data_w)[w];
											cls_conf = SIGMOID(cls_conf);

											xywhscore[4] = xywhscore[4] * cls_conf;
											if (xywhscore[4] > yolov5->conf_threshold) {
												xywhscore[0] = ((float *)x_data_w)[w];
												xywhscore[0] = SIGMOID(xywhscore[0]);
												xywhscore[0] = (xywhscore[0] * 2.0 - 0.5 + w) * stride_w;

												xywhscore[1] = ((float *)y_data_w)[w];
												xywhscore[1] = SIGMOID(xywhscore[1]);
												xywhscore[1] = (xywhscore[1] * 2.0 - 0.5 + h) * stride_h;
												x1y1x2y2score_in_class[c][valid_count_in_class[c] * 5 + 0] = xywhscore[0] - xywhscore[2] / 2.0;
												x1y1x2y2score_in_class[c][valid_count_in_class[c] * 5 + 1] = xywhscore[1] - xywhscore[3] / 2.0;
												x1y1x2y2score_in_class[c][valid_count_in_class[c] * 5 + 2] = xywhscore[0] + xywhscore[2] / 2.0;
												x1y1x2y2score_in_class[c][valid_count_in_class[c] * 5 + 3] = xywhscore[1] + xywhscore[3] / 2.0;
												x1y1x2y2score_in_class[c][valid_count_in_class[c] * 5 + 4] = xywhscore[4];
												valid_count_in_class[c]++;
												EA_LOG_DEBUG("%f %f %f %f %f\n", xywhscore[0], xywhscore[1], xywhscore[2], xywhscore[3], xywhscore[4]);
											}
										}
									}
								}
							}
						}
					}
				} else {
					for (h = 0; h < height[m]; h++) {
						x_data_w = x_data + h * pitch;
						y_data_w = y_data + h * pitch;
						w_data_w = w_data + h * pitch;
						h_data_w = h_data + h * pitch;
						x_data_w = x_data + h * pitch;
						box_conf_data_w = box_conf_data + h * pitch;
						for (w = 0; w < width[m]; w++) {
							xywhscore[4] = ((float *)box_conf_data_w)[w];
							xywhscore[4] = SIGMOID(xywhscore[4]);
							if (xywhscore[4] > yolov5->conf_threshold) {
								xywhscore[2] = ((float *)w_data_w)[w];
								xywhscore[2] = SIGMOID(xywhscore[2]);
								xywhscore[2] = pow(xywhscore[2] * 2.0, 2.0) * yolov5_anchors[m][a][0];
								if (xywhscore[2] > YOLOV5_MIN_WH && xywhscore[2] < YOLOV5_MAX_WH) {
									xywhscore[3] = ((float *)h_data_w)[w];
									xywhscore[3] = SIGMOID(xywhscore[3]);
									xywhscore[3] = pow(xywhscore[3] * 2.0, 2.0) * yolov5_anchors[m][a][1];
									if (xywhscore[3] > YOLOV5_MIN_WH && xywhscore[3] < YOLOV5_MAX_WH) {
										best_cls = 0;
										best_cls_conf = -FLT_MAX;
										for (c = 0; c < class_num; c++) {
											cls_conf_data = box_conf_data + shape[EA_H] * pitch + c * shape[EA_H] * pitch;
											cls_conf_data_w = cls_conf_data + h * pitch;
											cls_conf = ((float *)cls_conf_data_w)[w];
											if (best_cls_conf < cls_conf) {
												best_cls_conf = cls_conf;
												best_cls = c;
											}
										}

										c = best_cls;
										cls_conf = best_cls_conf;
										cls_conf = SIGMOID(cls_conf);

										xywhscore[4] = xywhscore[4] * cls_conf;
	#if 0	// the post process in python code doesn't have the following check on conf_threshold check.
										if (xywhscore[4] > yolov5->conf_threshold) {
	#endif
											xywhscore[0] = ((float *)x_data_w)[w];
											xywhscore[0] = SIGMOID(xywhscore[0]);
											xywhscore[0] = (xywhscore[0] * 2.0 - 0.5 + w) * stride_w;

											xywhscore[1] = ((float *)y_data_w)[w];
											xywhscore[1] = SIGMOID(xywhscore[1]);
											xywhscore[1] = (xywhscore[1] * 2.0 - 0.5 + h) * stride_h;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 5 + 0] = xywhscore[0] - xywhscore[2] / 2.0;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 5 + 1] = xywhscore[1] - xywhscore[3] / 2.0;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 5 + 2] = xywhscore[0] + xywhscore[2] / 2.0;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 5 + 3] = xywhscore[1] + xywhscore[3] / 2.0;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 5 + 4] = xywhscore[4];
											valid_count_in_class[c]++;
											EA_LOG_DEBUG("%f %f %f %f %f\n", xywhscore[0], xywhscore[1], xywhscore[2], xywhscore[3], xywhscore[4]);
	#if 0
										}
	#endif
									}
								}
							}
						}
					}
				}
			}
		}

		nms_x1y1x2y2score_in_class = (float **)malloc(class_num * sizeof(float *));
		RVAL_ASSERT(nms_x1y1x2y2score_in_class != NULL);
		for (c = 0; c < class_num; c++) {
			nms_x1y1x2y2score_in_class[c] = (float *)malloc(valid_count_in_class[c] * 5 * sizeof(float));
			RVAL_ASSERT(nms_x1y1x2y2score_in_class[c] != NULL);
		}

		RVAL_BREAK();

		nms_valid_count_in_class = (int *)malloc(class_num * sizeof(int));
		RVAL_ASSERT(valid_count_in_class != NULL);

		for (c = 0; c < class_num; c++) {
			RVAL_OK(ea_nms(x1y1x2y2score_in_class[c], NULL, 0, valid_count_in_class[c], yolov5->nms_threshold, 0/*use_iou_min*/, yolov5->top_k/*tok_k*/,
				nms_x1y1x2y2score_in_class[c], NULL, &nms_valid_count_in_class[c]));
		}

		RVAL_BREAK();

		result->valid_det_count = 0;
		for (c = 0; c < class_num; c++) {
			for (i = 0; i < nms_valid_count_in_class[c]; i++) {
				if (nms_x1y1x2y2score_in_class[c][i * 5 + 4] > yolov5->conf_threshold) {
					result->detections[result->valid_det_count].id = c;
					result->detections[result->valid_det_count].score = nms_x1y1x2y2score_in_class[c][i * 5 + 4];

					result->detections[result->valid_det_count].x_start =
						nms_x1y1x2y2score_in_class[c][i * 5 + 0] / ea_tensor_shape(yolov5->input_tensor)[EA_W];
					result->detections[result->valid_det_count].x_start = max(0.0, result->detections[result->valid_det_count].x_start);
					result->detections[result->valid_det_count].x_start = min(1.0, result->detections[result->valid_det_count].x_start);

					result->detections[result->valid_det_count].y_start =
						nms_x1y1x2y2score_in_class[c][i * 5 + 1] / ea_tensor_shape(yolov5->input_tensor)[EA_H];
					result->detections[result->valid_det_count].y_start = max(0.0, result->detections[result->valid_det_count].y_start);
					result->detections[result->valid_det_count].y_start = min(1.0, result->detections[result->valid_det_count].y_start);

					result->detections[result->valid_det_count].x_end =
						nms_x1y1x2y2score_in_class[c][i * 5 + 2] / ea_tensor_shape(yolov5->input_tensor)[EA_W];
					result->detections[result->valid_det_count].x_end =
						max(result->detections[result->valid_det_count].x_start, result->detections[result->valid_det_count].x_end);
					result->detections[result->valid_det_count].x_end =
						min(1.0, result->detections[result->valid_det_count].x_end);

					result->detections[result->valid_det_count].y_end =
						nms_x1y1x2y2score_in_class[c][i * 5 + 3] / ea_tensor_shape(yolov5->input_tensor)[EA_H];
					result->detections[result->valid_det_count].y_end =
						max(result->detections[result->valid_det_count].y_start, result->detections[result->valid_det_count].y_end);
					result->detections[result->valid_det_count].y_end =
						min(1.0, result->detections[result->valid_det_count].y_end);

					snprintf(result->detections[result->valid_det_count].label, sizeof(result->detections[result->valid_det_count].label),
						"%s", yolov5->labels[result->detections[result->valid_det_count].id]);
					EA_LOG_DEBUG("%f %f %f %f %f\n",
						nms_x1y1x2y2score_in_class[c][i * 5 + 0], nms_x1y1x2y2score_in_class[c][i * 5 + 1],
						nms_x1y1x2y2score_in_class[c][i * 5 + 2], nms_x1y1x2y2score_in_class[c][i * 5 + 3],
						nms_x1y1x2y2score_in_class[c][i * 5 + 4]);
					EA_LOG_DEBUG("%f %f %f %f %f\n",
						result->detections[result->valid_det_count].x_start, result->detections[result->valid_det_count].y_start,
						result->detections[result->valid_det_count].x_end, result->detections[result->valid_det_count].y_end,
						result->detections[result->valid_det_count].score);
					result->valid_det_count++;
					if (result->valid_det_count >= YOLOV5_MAX_OUT_NUM) {
						break;
					}
				}
			}

			if (result->valid_det_count >= YOLOV5_MAX_OUT_NUM) {
				break;
			}
		}
	} while (0);

	if (x1y1x2y2score_in_class) {
		for (c = 0; c < class_num; c++) {
			if (x1y1x2y2score_in_class[c]) {
				free(x1y1x2y2score_in_class[c]);
			}
		}

		free(x1y1x2y2score_in_class);
	}

	if (valid_count_in_class) {
		free(valid_count_in_class);
	}

	if (nms_x1y1x2y2score_in_class) {
		for (c = 0; c < class_num; c++) {
			if (nms_x1y1x2y2score_in_class[c]) {
				free(nms_x1y1x2y2score_in_class[c]);
			}
		}

		free(nms_x1y1x2y2score_in_class);
	}

	if (nms_valid_count_in_class) {
		free(nms_valid_count_in_class);
	}

	return rval;
}

int landmark_yolov5_arm_post_process(yolov5_t *yolov5, landmark_yolov5_result_s *result)
{
	int rval = 0;
	int class_num = ea_tensor_shape(yolov5->feature_map_tensors[0])[EA_C] / YOLOV5_ANCHOR_NUM - 13;
	int height[YOLOV5_FEATURE_MAP_NUM];
	int width[YOLOV5_FEATURE_MAP_NUM];
	int max_det_num_in_class = 0;
	uint8_t *feature_map_data;
	uint8_t *feature_map_data_in_anchor;
	uint8_t *x_data, *y_data, *w_data, *h_data, *box_conf_data, *cls_conf_data;
	uint8_t *x_data_w, *y_data_w, *w_data_w, *h_data_w, *box_conf_data_w, *cls_conf_data_w;
	uint8_t *landmarks_data1, *landmarks_data1_w,*landmarks_data2, *landmarks_data2_w,*landmarks_data3, *landmarks_data3_w,
			*landmarks_data4, *landmarks_data4_w,*landmarks_data5, *landmarks_data5_w,*landmarks_data6, *landmarks_data6_w,
			*landmarks_data7, *landmarks_data7_w,*landmarks_data8, *landmarks_data8_w;
	const size_t *shape;
	size_t pitch;
	int stride_w, stride_h;
	float xywhscore[5];
	float cls_conf;
	float **x1y1x2y2score_in_class = NULL;
	int *valid_count_in_class = NULL;
	float **nms_x1y1x2y2score_in_class = NULL;
	int *nms_valid_count_in_class = NULL;
	int i, m, a, c, h, w;

	do {
		for (m = 0; m < YOLOV5_FEATURE_MAP_NUM; m++) {
			RVAL_OK(ea_tensor_sync_cache(yolov5->feature_map_tensors[m], EA_VP, EA_CPU));
			height[m] = ea_tensor_shape(yolov5->feature_map_tensors[m])[EA_H];
			width[m] = ea_tensor_shape(yolov5->feature_map_tensors[m])[EA_W];
			max_det_num_in_class += height[m] * width[m] * YOLOV5_ANCHOR_NUM;
		}

		RVAL_BREAK();

		x1y1x2y2score_in_class = (float **)malloc(class_num * sizeof(float *));
		RVAL_ASSERT(x1y1x2y2score_in_class != NULL);
		for (c = 0; c < class_num; c++) {
			x1y1x2y2score_in_class[c] = (float *)malloc(max_det_num_in_class * 13 * sizeof(float));
			RVAL_ASSERT(x1y1x2y2score_in_class[c] != NULL);
		}

		RVAL_BREAK();

		valid_count_in_class = (int *)malloc(class_num * sizeof(int));
		RVAL_ASSERT(valid_count_in_class != NULL);
		memset(valid_count_in_class, 0, class_num * sizeof(int));

		for (m = 0; m < YOLOV5_FEATURE_MAP_NUM; m++) {
			feature_map_data = (uint8_t *)ea_tensor_data(yolov5->feature_map_tensors[m]);
			shape = ea_tensor_shape(yolov5->feature_map_tensors[m]);
			pitch = ea_tensor_pitch(yolov5->feature_map_tensors[m]);
			stride_w = ea_tensor_shape(yolov5->input_tensor)[EA_W] / shape[EA_W];
			stride_h = ea_tensor_shape(yolov5->input_tensor)[EA_H] / shape[EA_H];
			for (a = 0; a < YOLOV5_ANCHOR_NUM; a++) {
				feature_map_data_in_anchor = feature_map_data + shape[EA_H] * pitch * (class_num + 13) * a;
				x_data = feature_map_data_in_anchor;
				y_data = x_data + shape[EA_H] * pitch;
				w_data = y_data + shape[EA_H] * pitch;
				h_data = w_data + shape[EA_H] * pitch;
				box_conf_data = h_data + shape[EA_H] * pitch;
				landmarks_data1 = box_conf_data + shape[EA_H] * pitch;
				landmarks_data2 = landmarks_data1 + shape[EA_H] * pitch;
				landmarks_data3 = landmarks_data2 + shape[EA_H] * pitch;
				landmarks_data4 = landmarks_data3 + shape[EA_H] * pitch;
				landmarks_data5 = landmarks_data4 + shape[EA_H] * pitch;
				landmarks_data6 = landmarks_data5 + shape[EA_H] * pitch;
				landmarks_data7 = landmarks_data6 + shape[EA_H] * pitch;
				landmarks_data8 = landmarks_data7 + shape[EA_H] * pitch;
				for (c = 0; c < class_num; c++) {
					cls_conf_data = box_conf_data + 9*shape[EA_H] * pitch + c * shape[EA_H] * pitch;
					for (h = 0; h < height[m]; h++) {
						x_data_w = x_data + h * pitch;
						y_data_w = y_data + h * pitch;
						w_data_w = w_data + h * pitch;
						h_data_w = h_data + h * pitch;
						
						box_conf_data_w = box_conf_data + h * pitch;
						landmarks_data1_w = landmarks_data1 + h * pitch;
						landmarks_data2_w = landmarks_data2 + h * pitch;
						landmarks_data3_w = landmarks_data3 + h * pitch;
						landmarks_data4_w = landmarks_data4 + h * pitch;
						landmarks_data5_w = landmarks_data5 + h * pitch;
						landmarks_data6_w = landmarks_data6 + h * pitch;
						landmarks_data7_w = landmarks_data7 + h * pitch;
						landmarks_data8_w = landmarks_data8 + h * pitch;
						cls_conf_data_w = cls_conf_data + h * pitch;

						for (w = 0; w < width[m]; w++) {
							xywhscore[4] = ((float *)box_conf_data_w)[w];
							xywhscore[4] = SIGMOID(xywhscore[4]);
							if (xywhscore[4] > yolov5->conf_threshold) {
								xywhscore[2] = ((float *)w_data_w)[w];
								xywhscore[2] = SIGMOID(xywhscore[2]);
								xywhscore[2] = pow(xywhscore[2] * 2.0, 2.0) * landmark_yolov5_anchors[m][a][0];
								if (xywhscore[2] > YOLOV5_MIN_WH && xywhscore[2] < YOLOV5_MAX_WH) {
									xywhscore[3] = ((float *)h_data_w)[w];
									xywhscore[3] = SIGMOID(xywhscore[3]);
									xywhscore[3] = pow(xywhscore[3] * 2.0, 2.0) * landmark_yolov5_anchors[m][a][1];
									if (xywhscore[3] > YOLOV5_MIN_WH && xywhscore[3] < YOLOV5_MAX_WH) {
										cls_conf = ((float *)cls_conf_data_w)[w];
										cls_conf = SIGMOID(cls_conf);

										xywhscore[4] = xywhscore[4] * cls_conf;
										if (xywhscore[4] > yolov5->conf_threshold) {
											xywhscore[0] = ((float *)x_data_w)[w];
											xywhscore[0] = SIGMOID(xywhscore[0]);
											xywhscore[0] = (xywhscore[0] * 2.0 - 0.5 + w) * stride_w;

											xywhscore[1] = ((float *)y_data_w)[w];
											xywhscore[1] = SIGMOID(xywhscore[1]);
											xywhscore[1] = (xywhscore[1] * 2.0 - 0.5 + h) * stride_h;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 0] = xywhscore[0] - xywhscore[2] / 2.0;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 1] = xywhscore[1] - xywhscore[3] / 2.0;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 2] = xywhscore[0] + xywhscore[2] / 2.0;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 3] = xywhscore[1] + xywhscore[3] / 2.0;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 4] = xywhscore[4];
											//landmark
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 5] = ((float *)landmarks_data1_w)[w] * landmark_yolov5_anchors[m][a][0] + w * stride_w;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 6] = ((float *)landmarks_data2_w)[w] * landmark_yolov5_anchors[m][a][1] + h * stride_h;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 7] = ((float *)landmarks_data3_w)[w] * landmark_yolov5_anchors[m][a][0] + w * stride_w;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 8] = ((float *)landmarks_data4_w)[w] * landmark_yolov5_anchors[m][a][1] + h * stride_h;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 9] = ((float *)landmarks_data5_w)[w] * landmark_yolov5_anchors[m][a][0] + w * stride_w;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 10] = ((float *)landmarks_data6_w)[w] * landmark_yolov5_anchors[m][a][1] + h * stride_h;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 11] = ((float *)landmarks_data7_w)[w] * landmark_yolov5_anchors[m][a][0] + w * stride_w;
											x1y1x2y2score_in_class[c][valid_count_in_class[c] * 13 + 12] = ((float *)landmarks_data8_w)[w] * landmark_yolov5_anchors[m][a][1] + h * stride_h;
											
				
											valid_count_in_class[c]++;
											//EA_LOG_DEBUG("%f %f %f %f %f\n", xywhscore[0], xywhscore[1], xywhscore[2], xywhscore[3], xywhscore[4]);
										}
									}
								}
							}
						}
					}
				} 
			}
		}

		nms_x1y1x2y2score_in_class = (float **)malloc(class_num * sizeof(float *));
		RVAL_ASSERT(nms_x1y1x2y2score_in_class != NULL);
		for (c = 0; c < class_num; c++) {
			nms_x1y1x2y2score_in_class[c] = (float *)malloc(valid_count_in_class[c] * 13 * sizeof(float));
			RVAL_ASSERT(nms_x1y1x2y2score_in_class[c] != NULL);
		}

		RVAL_BREAK();

		nms_valid_count_in_class = (int *)malloc(class_num * sizeof(int));
		RVAL_ASSERT(valid_count_in_class != NULL);

		for (c = 0; c < class_num; c++) {
			RVAL_OK(landmark_nms(x1y1x2y2score_in_class[c], NULL, 0, valid_count_in_class[c], yolov5->nms_threshold, 0/*use_iou_min*/, yolov5->top_k/*tok_k*/,
				    nms_x1y1x2y2score_in_class[c], NULL, &nms_valid_count_in_class[c]));
		}

		RVAL_BREAK();

		result->valid_det_count = 0;
		for (c = 0; c < class_num; c++) {
			for (i = 0; i < nms_valid_count_in_class[c]; i++) {
				if (nms_x1y1x2y2score_in_class[c][i * 13 + 4] > yolov5->conf_threshold) {
					//score
					result->detections[result->valid_det_count].id = c;
					result->detections[result->valid_det_count].score = nms_x1y1x2y2score_in_class[c][i * 13 + 4];
					//bbox
					result->detections[result->valid_det_count].x_start =
						nms_x1y1x2y2score_in_class[c][i * 13 + 0] / ea_tensor_shape(yolov5->input_tensor)[EA_W];
					result->detections[result->valid_det_count].x_start = max(0.0, result->detections[result->valid_det_count].x_start);
					result->detections[result->valid_det_count].x_start = min(1.0, result->detections[result->valid_det_count].x_start);

					result->detections[result->valid_det_count].y_start =
						nms_x1y1x2y2score_in_class[c][i * 13 + 1] / ea_tensor_shape(yolov5->input_tensor)[EA_H];
					result->detections[result->valid_det_count].y_start = max(0.0, result->detections[result->valid_det_count].y_start);
					result->detections[result->valid_det_count].y_start = min(1.0, result->detections[result->valid_det_count].y_start);

					result->detections[result->valid_det_count].x_end =
						nms_x1y1x2y2score_in_class[c][i * 13 + 2] / ea_tensor_shape(yolov5->input_tensor)[EA_W];
					result->detections[result->valid_det_count].x_end =
						max(result->detections[result->valid_det_count].x_start, result->detections[result->valid_det_count].x_end);
					result->detections[result->valid_det_count].x_end =
						min(1.0, result->detections[result->valid_det_count].x_end);

					result->detections[result->valid_det_count].y_end =
						nms_x1y1x2y2score_in_class[c][i * 13 + 3] / ea_tensor_shape(yolov5->input_tensor)[EA_H];
					result->detections[result->valid_det_count].y_end =
						max(result->detections[result->valid_det_count].y_start, result->detections[result->valid_det_count].y_end);
					result->detections[result->valid_det_count].y_end =
						min(1.0, result->detections[result->valid_det_count].y_end);
					//landmark
					result->detections[result->valid_det_count].p1_x =
						nms_x1y1x2y2score_in_class[c][i * 13 + 5] / ea_tensor_shape(yolov5->input_tensor)[EA_W];
					result->detections[result->valid_det_count].p1_x = max(0.0, result->detections[result->valid_det_count].p1_x);
					result->detections[result->valid_det_count].p1_x = min(1.0, result->detections[result->valid_det_count].p1_x);

					result->detections[result->valid_det_count].p1_y =
						nms_x1y1x2y2score_in_class[c][i * 13 + 6] / ea_tensor_shape(yolov5->input_tensor)[EA_H];
					result->detections[result->valid_det_count].p1_y = max(0.0, result->detections[result->valid_det_count].p1_y);
					result->detections[result->valid_det_count].p1_y = min(1.0, result->detections[result->valid_det_count].p1_y);

					result->detections[result->valid_det_count].p2_x =
						nms_x1y1x2y2score_in_class[c][i * 13 + 7] / ea_tensor_shape(yolov5->input_tensor)[EA_W];
					result->detections[result->valid_det_count].p2_x = max(0.0, result->detections[result->valid_det_count].p2_x);
					result->detections[result->valid_det_count].p2_x = min(1.0, result->detections[result->valid_det_count].p2_x);

					result->detections[result->valid_det_count].p2_y =
						nms_x1y1x2y2score_in_class[c][i * 13 + 8] / ea_tensor_shape(yolov5->input_tensor)[EA_H];
					result->detections[result->valid_det_count].p2_y = max(0.0, result->detections[result->valid_det_count].p2_y);
					result->detections[result->valid_det_count].p2_y = min(1.0, result->detections[result->valid_det_count].p2_y);

					result->detections[result->valid_det_count].p3_x =
						nms_x1y1x2y2score_in_class[c][i * 13 + 9] / ea_tensor_shape(yolov5->input_tensor)[EA_W];
					result->detections[result->valid_det_count].p3_x = max(0.0, result->detections[result->valid_det_count].p3_x);
					result->detections[result->valid_det_count].p3_x = min(1.0, result->detections[result->valid_det_count].p3_x);

					result->detections[result->valid_det_count].p3_y =
						nms_x1y1x2y2score_in_class[c][i * 13 + 10] / ea_tensor_shape(yolov5->input_tensor)[EA_H];
					result->detections[result->valid_det_count].p3_y = max(0.0, result->detections[result->valid_det_count].p3_y);
					result->detections[result->valid_det_count].p3_y = min(1.0, result->detections[result->valid_det_count].p3_y);

					result->detections[result->valid_det_count].p4_x =
						nms_x1y1x2y2score_in_class[c][i * 13 + 11] / ea_tensor_shape(yolov5->input_tensor)[EA_W];
					result->detections[result->valid_det_count].p4_x = max(0.0, result->detections[result->valid_det_count].p4_x);
					result->detections[result->valid_det_count].p4_x = min(1.0, result->detections[result->valid_det_count].p4_x);

					result->detections[result->valid_det_count].p4_y =
						nms_x1y1x2y2score_in_class[c][i * 13 + 12] / ea_tensor_shape(yolov5->input_tensor)[EA_H];
					result->detections[result->valid_det_count].p4_y = max(0.0, result->detections[result->valid_det_count].p4_y);
					result->detections[result->valid_det_count].p4_y = min(1.0, result->detections[result->valid_det_count].p4_y);

					snprintf(result->detections[result->valid_det_count].label, sizeof(result->detections[result->valid_det_count].label),
						"%s", yolov5->labels[result->detections[result->valid_det_count].id]);
					
					EA_LOG_DEBUG("bbox:%f %f %f %f %f\n",
						nms_x1y1x2y2score_in_class[c][i * 13 + 0], nms_x1y1x2y2score_in_class[c][i * 13 + 1],
						nms_x1y1x2y2score_in_class[c][i * 13 + 2], nms_x1y1x2y2score_in_class[c][i * 13 + 3],
						nms_x1y1x2y2score_in_class[c][i * 13 + 4]);
					EA_LOG_DEBUG("bbox_norm:%f %f %f %f %f\n",
						result->detections[result->valid_det_count].x_start, result->detections[result->valid_det_count].y_start,
						result->detections[result->valid_det_count].x_end, result->detections[result->valid_det_count].y_end,
						result->detections[result->valid_det_count].score);
					EA_LOG_DEBUG("landmarks:%f %f %f %f %f %f %f %f\n",
						nms_x1y1x2y2score_in_class[c][i * 13 + 5], nms_x1y1x2y2score_in_class[c][i * 13 + 6],
						nms_x1y1x2y2score_in_class[c][i * 13 + 7], nms_x1y1x2y2score_in_class[c][i * 13 + 8],
						nms_x1y1x2y2score_in_class[c][i * 13 + 9], nms_x1y1x2y2score_in_class[c][i * 13 + 10],
						nms_x1y1x2y2score_in_class[c][i * 13 + 11], nms_x1y1x2y2score_in_class[c][i * 13 + 12]);
					EA_LOG_DEBUG("landmarks_norm%f %f %f %f %f %f %f %f\n",
						result->detections[result->valid_det_count].p1_x, result->detections[result->valid_det_count].p1_y,
						result->detections[result->valid_det_count].p2_x, result->detections[result->valid_det_count].p2_y,
						result->detections[result->valid_det_count].p3_x, result->detections[result->valid_det_count].p3_y,
						result->detections[result->valid_det_count].p4_x, result->detections[result->valid_det_count].p4_y);
					
					result->valid_det_count++;
					if (result->valid_det_count >= YOLOV5_MAX_OUT_NUM) {
						break;
					}
				}
			}

			if (result->valid_det_count >= YOLOV5_MAX_OUT_NUM) {
				break;
			}
		}
	} while (0);

	if (x1y1x2y2score_in_class) {
		for (c = 0; c < class_num; c++) {
			if (x1y1x2y2score_in_class[c]) {
				free(x1y1x2y2score_in_class[c]);
			}
		}

		free(x1y1x2y2score_in_class);
	}

	if (valid_count_in_class) {
		free(valid_count_in_class);
	}

	if (nms_x1y1x2y2score_in_class) {
		for (c = 0; c < class_num; c++) {
			if (nms_x1y1x2y2score_in_class[c]) {
				free(nms_x1y1x2y2score_in_class[c]);
			}
		}

		free(nms_x1y1x2y2score_in_class);
	}

	if (nms_valid_count_in_class) {
		free(nms_valid_count_in_class);
	}

	return rval;
}

