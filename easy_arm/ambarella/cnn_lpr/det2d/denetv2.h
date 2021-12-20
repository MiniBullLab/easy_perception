#ifndef _DENETV2_H_
#define _DENETV2_H_

#include "cnn_lpr/common/common_process.h"

#include "yolov5.h"

#include <string>
#include <vector>

class DeNetV2{
public:
    DeNetV2();
    ~DeNetV2();
    void set_log(int debug_en = 0);
    int init(const std::string &modelPath, const std::vector<std::string> &inputName, 
             const std::vector<std::string> &outputName, 
             const int classNumber, const float threshold=0.3f);
    std::vector<std::vector<float>> run(ea_tensor_t *img_tensor);

private:
    float threshold;
    float nms_threshold;
    int top_k;
	int use_multi_cls;
    int classNumber;

    int log_level;
    yolov5_t yolov5_ctx;
};

#endif // _DENETV2_H_