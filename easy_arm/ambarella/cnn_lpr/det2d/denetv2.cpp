#include "denetv2.h"

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
    int inputCount = static_cast<int>(inputName.size());
    int outputCount = static_cast<int>(outputName.size());
    char ** inNames = new char*[inputCount];
    char ** outNames = new char*[outputCount];

    for(size_t i = 0; i < inputName.size(); i++){
        inNames[i] = new char[inputName[i].size() + 1];
        strcpy(inNames[i], inputName[i].c_str());
    }
    for(size_t i = 0; i < outputName.size(); i++){
        outNames[i] = new char[outputName[i].size() + 1];
        strcpy(outNames[i], outputName[i].c_str());
    }

    do {
		memset(&net_params, 0, sizeof(yolov5_params_t));
		net_params.log_level = log_level;
        net_params.feature_map_names[0] = outNames[0];
		net_params.feature_map_names[1] = outNames[1];
		net_params.feature_map_names[2] = outNames[2];

		net_params.model_path = modelPath.c_str();
		net_params.label_path = NULL;
		net_params.input_name = inNames[0];

		net_params.conf_threshold = threshold;
		net_params.nms_threshold = nms_threshold;
		net_params.keep_top_k = top_k;
		net_params.use_multi_cls = use_multi_cls;
		RVAL_OK(yolov5_init(&yolov5_ctx, &net_params));
	} while (0);

    this->classNumber = classNumber;
    this->threshold = threshold;

    for(size_t i = 0; i < inputName.size(); i++){
        delete [] inNames[i];
    }
    delete [] inNames;
    for(size_t i = 0; i < outputName.size(); i++){
        delete [] outNames[i];
    }
    delete [] outNames;

    return rval;
}

std::vector<std::vector<float>> DeNetV2::run(ea_tensor_t *img_tensor)
{
	int rval = 0;
	std::vector<std::vector<float>> results;
	yolov5_result_t yolov5_net_result;
	int width = ea_tensor_shape(img_tensor)[3];
	int height = ea_tensor_shape(img_tensor)[2];
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
		float xmin = yolov5_net_result.detections[i].x_end * width;
		float ymin = yolov5_net_result.detections[i].y_end * height;
		float xmax = yolov5_net_result.detections[i].x_start * width;
		float ymax = yolov5_net_result.detections[i].y_start * height;
		box.clear();
		box.push_back(xmin);
		box.push_back(ymin);
		box.push_back(xmax - xmin);
		box.push_back(ymax - ymin);
		box.push_back(yolov5_net_result.detections[i].score);
		box.push_back(yolov5_net_result.detections[i].id);
		results.push_back(box);
	}
	return results;
}