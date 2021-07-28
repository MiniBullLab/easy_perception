#include "cnn_runtime/rec_text/textnet.h"
#include "cnn_runtime/cnn_common/net_process.h"
#include "cnn_runtime/cnn_common/cnn_function.h"
#include "cnn_runtime/cnn_common/image_process.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>

const static int maxTextLength = 32;
const static int classNumber = 38;
// const static char characterSet[classNumber] = {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
//                                                'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
//                                                's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 
//                                                'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
//                                                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
//                                                'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', 
//                                                '7', '8', '9', '0', '!', '"', '#', '$', '%', '&', 
//                                                '\\', '\'', '(', ')', '*', '+', ',', '-', '.', '/', 
//                                                ':', ';', '<', '=', '>', '?', '@', '[', ']', '^', 
//                                                '_', '`', '{', '}', '~', '|', ' '};
const static char characterSet[classNumber] = {' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                                               'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                                               'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', 
                                               '6', '7', '8', '9', '0', ' '};

TextNet::TextNet()
{
    memset(&cavalry_ctx, 0, sizeof(cavalry_ctx_t));
    memset(&nnctrl_ctx, 0, sizeof(nnctrl_ctx_t));
    threshold = 0;
    textnetOutput = NULL;
}

TextNet::~TextNet()
{
    deinit_net_context(&nnctrl_ctx, &cavalry_ctx);
    DPRINT_NOTICE("mtcnn_deinit\n");
    if(textnetOutput != NULL)
    {
        delete[] textnetOutput;
        textnetOutput = NULL;
    }
}

int TextNet::init(const std::string &modelPath, const std::string &inputName, \
                  const std::string &outputName, const float threshold, const int charCount)
{
    int rval = 0;
    set_net_param(&nnctrl_ctx, modelPath.c_str(), \
                    inputName.c_str(), outputName.c_str());
    rval = cnn_init(&nnctrl_ctx, &cavalry_ctx);
    this->charCount = charCount;
    this->threshold = threshold;
    this->textnetOutput = new float[maxTextLength * classNumber];

    return rval;
}

std::string TextNet::run(const cv::Mat &srcImage)
{
    int count = 0;
    std::string result = "";
    float max_score = 0.0f;
    int pre_max_index = 0;
    int max_index = 0;
    float *tempOutput[1] = {NULL};
    cv::Mat filterMat;
    cv::medianBlur(srcImage, filterMat, 5);
    // cv::bilateralFilter(srcImage, filterMat, 10, 10 * 2, 10 / 2);
    preprocess(&nnctrl_ctx, filterMat, 2);
    cnn_run(&nnctrl_ctx, tempOutput, 1);
    int output_c = nnctrl_ctx.net.net_out.out_desc[0].dim.depth;
    int output_h = nnctrl_ctx.net.net_out.out_desc[0].dim.height;
    int output_w = nnctrl_ctx.net.net_out.out_desc[0].dim.width;
    int output_p = nnctrl_ctx.net.net_out.out_desc[0].dim.pitch / 4;

    std::cout << "output size: " << "--output_c: " << output_c << "--output_h: " << output_h << "--output_w: " \
                                  << output_w << "--output_p: " << output_p << "--" << std::endl;

    for (int h = 0; h < output_h; h++)
    {
        memcpy(textnetOutput + h * output_w, tempOutput[0] + h * output_p, output_w * sizeof(float));
    }
    // std::ofstream ouF;
    // ouF.open("./score.bin", std::ofstream::binary);
    // ouF.write(reinterpret_cast<const char*>(textnetOutput), sizeof(float) * maxTextLength * classNumber);
    // ouF.close();
    for (int row = 0; row < maxTextLength; row++) {
        float* output = textnetOutput + row * classNumber;
        softmax(classNumber, output);
        max_score = this->threshold;
        max_index = 0;
        for (int col = 0; col < classNumber; col++) {
            if (output[col] > max_score) {
                max_score = output[col];
                max_index = col;
            }
        }
        if((max_index > 0) && !(row > 0 && max_index == pre_max_index))
        {
            if(characterSet[max_index] != ' ')
            {
                // std::cout << max_score << std::endl;
                result += characterSet[max_index];
                count++;
            }
        }
        pre_max_index = max_index;
    }
    return result;
}