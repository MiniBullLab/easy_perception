#ifndef PATCHCORE_POSTPORCESS_H
#define PATCHCORE_POSTPORCESS_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>
#include <fstream>

cv::Mat train_embedding_process(const std::string &embedding_file, const int output_channel);

cv::Mat reshape_embedding(const float *output,
                          const int out_channel, \ 
                          const int out_height, \
                          const int out_width);



#endif // POESNET_POSTPORCESS_H