#include "denetv2.h"
#include <iostream>

DeNetV2::DeNetV2()
{
    threshold = 0;
    nms_threshold = 0.3f;
    top_k = 100;
	use_multi_cls = 0;

	log_level = 0;
}

DeNetV2::~DeNetV2()
{
    yolov5_deinit(&yolov5_ctx);
}

void DeNetV2::set_log(int debug_en)
{
	log_level = debug_en;
}

int DeNetV2::init(const std::string &modelPath, const std::vector<std::string> &inputName, 
                  const std::vector<std::string> &outputName, 
                  const int classNumber, const float threshold)
{
    int rval = 0;
	yolov5_params_t net_params;
    do {
		memset(&net_params, 0, sizeof(yolov5_params_t));
		net_params.log_level = log_level;
        net_params.feature_map_names[0] = outputName[0].c_str();
		net_params.feature_map_names[1] = outputName[1].c_str();
		net_params.feature_map_names[2] = outputName[2].c_str();

		net_params.model_path = modelPath.c_str();
		net_params.input_name = inputName[0].c_str();
		net_params.label_path = NULL;

		net_params.conf_threshold = threshold;
		net_params.nms_threshold = nms_threshold;
		net_params.keep_top_k = top_k;
		net_params.use_multi_cls = use_multi_cls;
		RVAL_OK(yolov5_init(&yolov5_ctx, &net_params));
		// std::cout << "yolov5 size:" << ea_tensor_shape(yolov5_ctx.input_tensor)[0] << " " << ea_tensor_shape(yolov5_ctx.input_tensor)[1] << " " << ea_tensor_shape(yolov5_ctx.input_tensor)[2] << " " << ea_tensor_shape(yolov5_ctx.input_tensor)[3] << std::endl;
	} while (0);

    this->classNumber = classNumber;
    this->threshold = threshold;

    return rval;
}

std::vector<std::vector<float>> DeNetV2::run(ea_tensor_t *img_tensor)
{
	int rval = 0;
	std::vector<std::vector<float>> results;
	yolov5_result_t yolov5_net_result;
	int width = ea_tensor_shape(img_tensor)[3];
	int height = ea_tensor_shape(img_tensor)[2];
	float box_width = 0;
	float box_height = 0;
	results.clear();
	do {
		memset(&yolov5_net_result, 0, sizeof(yolov5_result_t));
		RVAL_OK(ea_cvt_color_resize(img_tensor, yolov5_input(&yolov5_ctx), EA_COLOR_YUV2RGB_NV12, EA_VP));
		RVAL_OK(yolov5_vp_forward(&yolov5_ctx));
		RVAL_OK(yolov5_arm_post_process(&yolov5_ctx, &yolov5_net_result));
	} while (0);
	for(int i = 0; i < yolov5_net_result.valid_det_count; i++) 
	{
		std::vector<float> box;
		float xmin = yolov5_net_result.detections[i].x_start * width;
		float ymin = yolov5_net_result.detections[i].y_start * height;
		float xmax = yolov5_net_result.detections[i].x_end * width;
		float ymax = yolov5_net_result.detections[i].y_end * height;
		if(xmin < 0)
			xmin = 1;
		if(ymin < 0)
			ymin = 1;
		if(xmax >= width)
			xmax = width - 1;
		if(ymax >= height)
			ymax = height -1;
		box_width = xmax - xmin;
		box_height = ymax - ymin;
		if(box_width < 10 || box_height < 10)
			continue;
		box.clear();
		box.push_back(xmin);
		box.push_back(ymin);
		box.push_back(box_width);
		box.push_back(box_height);
		box.push_back(yolov5_net_result.detections[i].id);
		box.push_back(yolov5_net_result.detections[i].score);
		results.push_back(box);
	}
	return results;
}