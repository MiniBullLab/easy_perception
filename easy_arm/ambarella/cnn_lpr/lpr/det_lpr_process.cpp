#include "det_lpr_process.h"

#define DEFAULT_SSD_CLASS_NUM		(2) /* For license and background */
#define DEFAULT_BACKGROUND_ID		(0)
#define DEFAULT_KEEP_TOP_K			(50)
#define DEFAULT_NMS_TOP_K			(100)
#define DEFAULT_NMS_THRES			(0.45f)
#define DEFAULT_SSD_CONF_THRES		(0.3f)
#define DEFAULT_YOLOV5_CONF_THRES   (0.6f)

const static std::string ssd_model_path = "/data/lpr/mobilenetv1_ssd_cavalry.bin";
const static std::string ssd_priorbox_path = "/data/lpr/lpr_priorbox_fp32.bin";
const static std::vector<std::string> ssd_input_name = {"data"};
const static std::vector<std::string> ssd_output_name = {"mbox_loc", "mbox_conf_flatten"};

const static std::string yolov5_model_path = "/data/lpr/yolov5plate_cavalry.bin";
const static std::vector<std::string> yolov5_input_name = {"data"};
const static std::vector<std::string> yolov5_output_name = {"989", "969", "949"};

void upscale_normalized_rectangle(float x_min, float y_min,
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


int init_ssd(SSD_ctx_t *SSD_ctx, global_control_param_t *G_param,
	uint32_t buffer_h, uint32_t buffer_w)
{
	int rval = 0;
	ssd_net_params_t ssd_net_params;
	ssd_tf_scale_factors_t scale_factors;
	LOG(INFO) << "init_ssd";
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
		ssd_net_params.conf_threshold = DEFAULT_SSD_CONF_THRES;
		ssd_net_params.keep_top_k = DEFAULT_KEEP_TOP_K;
		ssd_net_params.nms_threshold = DEFAULT_NMS_THRES;
		ssd_net_params.nms_top_k = DEFAULT_NMS_TOP_K;
		ssd_net_params.background_label_id = DEFAULT_BACKGROUND_ID;
		ssd_net_params.unnormalized = 0;
		ssd_net_params.class_num = DEFAULT_SSD_CLASS_NUM;
		ssd_net_params.priority = SSD_PRIORITY;
		ssd_net_params.debug_en = (G_param->debug_en >= INFO_LEVEL);
		ssd_net_params.nnctrl_print_time = (G_param->verbose);
		ssd_net_params.abort_if_preempted = G_param->abort_if_preempted;
		scale_factors.center_x_scale = 0;
		scale_factors.center_y_scale = 0;
		scale_factors.height_scale = 0;
		scale_factors.width_scale = 0;
		ssd_net_params.scale_factors = &scale_factors;
		RVAL_OK(ssd_net_init(&ssd_net_params, &SSD_ctx->ssd_net_ctx,
			&SSD_ctx->net_input, &SSD_ctx->vp_result_info));
	} while (0);
	LOG(INFO) << "init_ssd success";
	return rval;
}

int init_yolov5(yolov5_t *live_ctx, global_control_param_t *G_param)
{
	int rval = 0;
	yolov5_params_t net_params;

	do {
		memset(&net_params, 0, sizeof(yolov5_params_t));

		net_params.log_level = G_param->debug_en;
		net_params.feature_map_names[0] = yolov5_output_name[0].c_str();
		net_params.feature_map_names[1] = yolov5_output_name[1].c_str();;
		net_params.feature_map_names[2] = yolov5_output_name[2].c_str();
		net_params.input_name = yolov5_input_name[0].c_str();
		net_params.model_path = yolov5_model_path.c_str();
		net_params.label_path = NULL;

		net_params.conf_threshold = DEFAULT_YOLOV5_CONF_THRES;
		net_params.nms_threshold = 0.5f;
		net_params.keep_top_k = DEFAULT_NMS_TOP_K;
		RVAL_OK(yolov5_init(live_ctx, &net_params));

		// std::cout << "input size:" << ea_tensor_shape(live_ctx->input_tensor)[0] << " " << ea_tensor_shape(live_ctx->input_tensor)[1] << " " << ea_tensor_shape(live_ctx->input_tensor)[2] << " " << ea_tensor_shape(live_ctx->input_tensor)[3] << " " << ea_tensor_pitch(live_ctx->input_tensor) << std::endl;
		// for (int i = 0; i < YOLOV5_FEATURE_MAP_NUM; i++) {
		// 	std::cout << "output size:" << ea_tensor_shape(live_ctx->feature_map_tensors[i])[0] << " " << ea_tensor_shape(live_ctx->feature_map_tensors[i])[1] << " " << ea_tensor_shape(live_ctx->feature_map_tensors[i])[2] << " " << ea_tensor_shape(live_ctx->feature_map_tensors[i])[3] << " " << ea_tensor_pitch(live_ctx->feature_map_tensors[i]) << std::endl;
		// }

	} while (0);

	return rval;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float left = max(x1, x2);
    float right = min(x1 + w1, x2 + w2);
    return right - left;
}

float cal_iou(std::vector<float> box, std::vector<float> truth)
{
    float w = overlap(box[0], box[2], truth[0], truth[2]);
    float h = overlap(box[1], box[3], truth[1], truth[3]);
    if(w < 0 || h < 0) 
		return 0;

    float inter_area = w * h;
    // float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
	float union_area = truth[2] * truth[3];
    return inter_area * 1.0f / union_area;
}
