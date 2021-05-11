#ifndef POESNET_POSTPORCESS_H
#define POESNET_POSTPORCESS_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

struct KeyPoint{
    KeyPoint(cv::Point point,float probability){
        this->id = -1;
        this->point = point;
        this->probability = probability;
    }

    int id;
    cv::Point point;
    float probability;
};

struct ValidPair{
    ValidPair(int aId,int bId,float score){
        this->aId = aId;
        this->bId = bId;
        this->score = score;
    }

    int aId;
    int bId;
    float score;
};

// void splitNetOutputBlobToParts(float *netOutputBlob, const cv::Size& targetSize, std::vector<cv::Mat>& netOutputParts);

// void splitNetOutputBlobToParts(float *netOutputBlob0, float *netOutputBlob1,
//                                const cv::Size& targetSize, std::vector<cv::Mat>& netOutputParts);

void getPostnetResult(const std::vector<cv::Mat>& netOutputParts, std::vector<std::vector<KeyPoint>> &result);

#endif // POESNET_POSTPORCESS_H
