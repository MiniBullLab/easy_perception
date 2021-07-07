#pragma once

//opencv
#include <opencv2/core.hpp>

#include "inference/common/cnn_data_structure.h"


class TextNet{
public:
    typedef struct textnet_ctx_s {
        cavalry_ctx_t cavalry_ctx;
        nnctrl_ctx_t nnctrl_ctx;
    }textnet_ctx_s;

public:
    TextNet();
    ~TextNet();
    int init(const std::string &modelPath, const std::string &inputName, const std::string &outputName);
    std::string run(const cv::Mat &src_img);

private:
    std::string modelPath;
    std::string inputName;
    std::string outputName;
    textnet_ctx_s textnet_ctx;
    float *textnetOutput;
};