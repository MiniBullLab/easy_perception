#include "../lpr/Pipeline.h"


using namespace std;


int TEST_PIPELINE(){

    // pr::PipelinePR prc("./model/cascade.xml","./model/LPR.mlz");

    pr::PipelinePR prc("/easy_data/easy_perception/easy_ai/test_data/lpr_model/cascade.xml",
                      "/easy_data/easy_perception/easy_ai/test_data/lpr_model/HorizonalFinemapping.prototxt",
                      "/easy_data/easy_perception/easy_ai/test_data/lpr_model/HorizonalFinemapping.caffemodel",
                      "/easy_data/easy_perception/easy_ai/test_data/lpr_model/Segmentation.prototxt",
                      "/easy_data/easy_perception/easy_ai/test_data/lpr_model/Segmentation.caffemodel",
                      "/easy_data/easy_perception/easy_ai/test_data/lpr_model/lpr.prototxt",
                      "/easy_data/easy_perception/easy_ai/test_data/lpr_model/lpr.caffemodel",
                       "/easy_data/easy_perception/easy_ai/test_data/lpr_model/test.prototxt",
                       "/easy_data/easy_perception/easy_ai/test_data/lpr_model/test.caffemodel"
                    );

    cv::Mat image = cv::imread("/easy_data/image_1.jpg");

    std::vector<pr::PlateInfo> res = prc.RunPiplineAsImage(image,pr::SEGMENTATION_FREE_METHOD,45,500,3);
	int num = 0;


	std::vector<std::string> color{ "BLUE", "YELLOW", "WHITE", "GREEN", "UNKNOWN", "BLACK" };

    for(auto st:res) {
        if(st.confidence>0.8 ) {

            cv::Rect roi(0,36*num,136,36);
            cv::Mat roi_image = image(roi);
            st.getPlateImage().copyTo(roi_image);

            num++;

            std::cout << st.getPlateName() << " " << st.confidence << std::endl;
            cv::Rect region = st.getPlateRect();
            cv::rectangle(image,cv::Point(region.x,region.y),cv::Point(region.x+region.width,region.y+region.height),cv::Scalar(255,255,0),2);
			std::cout << color[st.getPlateType()] << std::endl;
			
        }
    }
	cv::imwrite("test.png", image);


    cv::imshow("image",image);
    cv::waitKey(0);
	return 0;
}



int main()
{
	TEST_PIPELINE();

    return 0 ;
}