#include "PlateDetection.h"
#include "util.h"


namespace pr{


    PlateDetection::PlateDetection(std::string filename_cascade){
        cascade.load(filename_cascade);
    };


    void PlateDetection::plateDetectionRough(cv::Mat InputImage,std::vector<pr::PlateInfo>  &plateInfos,int min_w,int max_w){

        cv::Mat processImage;

         cv::cvtColor(InputImage,processImage,cv::COLOR_BGR2GRAY);


        std::vector<cv::Rect> platesRegions;
//        std::vector<PlateInfo> plates;
        cv::Size minSize(min_w,min_w/4);
        cv::Size maxSize(max_w,max_w/4);
//        cv::imshow("input",InputImage);
//                cv::waitKey(0);
        cascade.detectMultiScale( processImage, platesRegions,
                                  1.1, 3, cv::CASCADE_SCALE_IMAGE,minSize,maxSize);

		nms(platesRegions, 0.4);
        for(auto plate:platesRegions)
        {
            // extend rects
//            x -= w * 0.14
//            w += w * 0.28
//            y -= h * 0.6
//            h += h * 1.1;

            cv::Mat plateForColorJudge = util::cropFromImage(InputImage,plate);
//            cv::Mat srcGray;
            util::Color plateColor = util::getPlateType(plateForColorJudge, true);

            int zeroadd_w  = static_cast<int>(plate.width*0.3);
            int zeroadd_h = static_cast<int>(plate.height*0.3);
            int zeroadd_x = static_cast<int>(plate.width*0.15);
            int zeroadd_y = static_cast<int>(plate.height*0.15);




            plate.x-=zeroadd_x;
            plate.y-=zeroadd_y;
            plate.height += zeroadd_h;
            plate.width += zeroadd_w;
            cv::Mat plateImage = util::cropFromImage(InputImage,plate);
            PlateInfo plateInfo(plateImage,plate);
            plateInfo.plateColor = (PlateColor)plateColor;
            plateInfos.push_back(plateInfo);

        }
    }

	double PlateDetection::IOU(const cv::Rect& r1, const cv::Rect& r2)
	{
		int x1 = std::max(r1.x, r2.x);
		int y1 = std::max(r1.y, r2.y);
		int x2 = std::min(r1.x + r1.width, r2.x + r2.width);
		int y2 = std::min(r1.y + r1.height, r2.y + r2.height);
		int w = std::max(0, (x2 - x1 + 1));
		int h = std::max(0, (y2 - y1 + 1));
		double inter = w * h;
		double o = inter / (r1.area() + r2.area() - inter);
		return (o >= 0) ? o : 0;
	}

	void PlateDetection::nms(std::vector<cv::Rect>& proposals, const double nms_threshold)
	{
		std::vector<int> scores;
		for (auto i : proposals) scores.push_back(i.area());

		std::vector<int> index;
		for (int i = 0; i < scores.size(); ++i){
			index.push_back(i);
		}

		std::sort(index.begin(), index.end(), [&](int a, int b){
			return scores[a] > scores[b];
		});

		std::vector<bool> del(scores.size(), false);
		for (size_t i = 0; i < index.size(); i++){
			if (!del[index[i]]){
				for (size_t j = i + 1; j < index.size(); j++){
					if (IOU(proposals[index[i]], proposals[index[j]]) > nms_threshold){
						del[index[j]] = true;
					}
				}
			}
		}

		std::vector<cv::Rect> new_proposals;
		for (const auto i : index){
			if (!del[i]) new_proposals.push_back(proposals[i]);
		}
		proposals = new_proposals;
	}


//    std::vector<pr::PlateInfo> PlateDetection::plateDetectionRough(cv::Mat InputImage,cv::Rect roi,int min_w,int max_w){
//        cv::Mat roi_region = util::cropFromImage(InputImage,roi);
//        return plateDetectionRough(roi_region,min_w,max_w);
//    }




}//namespace pr
