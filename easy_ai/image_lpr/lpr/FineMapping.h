

#ifndef SWIFTPR_FINEMAPPING_H
#define SWIFTPR_FINEMAPPING_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include "ModelLoader.h"

namespace pr{
    class FineMapping{
    public:
        FineMapping();


        FineMapping(std::string prototxt,std::string caffemodel);
        FineMapping(ModelLoader* modeloader);

        static cv::Mat FineMappingVertical(cv::Mat InputProposal,int sliceNum=15,int upper=0,int lower=-50,int windows_size=17,int paddingUpper=-1,int paddingLower= 1);
        cv::Mat FineMappingHorizon(cv::Mat FinedVertical,int leftPadding,int rightPadding);


    private:
        cv::dnn::Net net;

    };




}
#endif //SWIFTPR_FINEMAPPING_H
