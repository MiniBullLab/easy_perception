
#ifndef SWIFTPR_PLATEDETECTION_H
#define SWIFTPR_PLATEDETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "PlateInfo.h"

namespace pr{
    class PlateDetection{
    public:
        PlateDetection(std::string filename_cascade);
        PlateDetection();
        void LoadModel(std::string filename_cascade);
        void plateDetectionRough(cv::Mat InputImage,std::vector<pr::PlateInfo>  &plateInfos,int min_w=36,int max_w=800);
//        std::vector<pr::PlateInfo> plateDetectionRough(cv::Mat InputImage,int min_w= 60,int max_h = 400);


//        std::vector<pr::PlateInfo> plateDetectionRoughByMultiScaleEdge(cv::Mat InputImage);



    private:
        cv::CascadeClassifier cascade;
		double IOU(const cv::Rect& r1, const cv::Rect& r2);
		void nms(std::vector<cv::Rect>& proposals, const double nms_threshold);


    };

}// namespace pr

#endif //SWIFTPR_PLATEDETECTION_H
