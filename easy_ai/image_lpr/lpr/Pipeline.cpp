
#include "Pipeline.h"
namespace pr {
	std::vector<std::string> CH_PLATE_CODE;
	std::vector<std::string> loadLabel(std::string filename)
	{
		std::ifstream fin(filename, std::ios::in);
		char line[1024] = { 0 };
		std::vector<std::string> labels;

		while (fin.getline(line, sizeof(line)))
		{
			std::stringstream word(line);
			std::string string;
			word >> string;
			labels.push_back(string);
		}
		fin.clear();
		fin.close();
		return labels;

	}

    const int HorizontalPadding = 4;
    PipelinePR::PipelinePR(std::string detector_filename,
                           std::string finemapping_prototxt, std::string finemapping_caffemodel,
                           std::string segmentation_prototxt, std::string segmentation_caffemodel,
                           std::string charRecognization_proto, std::string charRecognization_caffemodel,
                           std::string segmentationfree_proto,std::string segmentationfree_caffemodel) {
        plateDetection = new PlateDetection(detector_filename);
        fineMapping = new FineMapping(finemapping_prototxt, finemapping_caffemodel);
        plateSegmentation = new PlateSegmentation(segmentation_prototxt, segmentation_caffemodel);
        generalRecognizer = new CNNRecognizer(charRecognization_proto, charRecognization_caffemodel);
        segmentationFreeRecognizer =  new SegmentationFreeRecognizer(segmentationfree_proto,segmentationfree_caffemodel);
		CH_PLATE_CODE = loadLabel("/easy_data/easy_perception/easy_ai/test_data/lpr_model/plate_code.txt");


    }

    PipelinePR::PipelinePR(std::string detector_filename, std::string model_compress) {
        plateDetection = new PlateDetection(detector_filename);
        modelLoader  = new ModelLoader(model_compress);
        fineMapping = new FineMapping(modelLoader);
        plateSegmentation = new PlateSegmentation(modelLoader);
        generalRecognizer = new CNNRecognizer(modelLoader);
        segmentationFreeRecognizer =  new SegmentationFreeRecognizer(modelLoader);
		CH_PLATE_CODE = loadLabel("/easy_data/easy_perception/easy_ai/test_data/lpr_model/plate_code.txt");

    }

    PipelinePR::~PipelinePR() {

        delete plateDetection;
        delete fineMapping;
        delete plateSegmentation;
        delete generalRecognizer;
        delete segmentationFreeRecognizer;
        delete modelLoader;

    }
	


    std::vector<PlateInfo> PipelinePR::RunPiplineAsImage(cv::Mat plateImage,int SegmentationMethod,int minPlateSize,int maxPlateSize,int PlateMargin){
        std::vector<PlateInfo> results;
        std::vector<pr::PlateInfo> plates;

        bool color_reverse_flag = 0 ;
        double t0= (double)cv::getCPUTickCount();
        plateDetection->plateDetectionRough(plateImage,plates,minPlateSize,maxPlateSize);
//  std::cout<<"detection:"<<((double)cv::getCPUTickCount() - t0 )*1000 / cv::getTickFrequency()<<"ms"<<std::endl;
        for (pr::PlateInfo plateinfo:plates) {

            cv::Mat image_finemapping = plateinfo.getPlateImage();
     /*       if(plateinfo.plateColor == YELLOW || plateinfo.plateColor == GREEN || plateinfo.plateColor == WHITE)
            {
                cv::bitwise_not(image_finemapping,image_finemapping);
                color_reverse_flag = 1;
            }*/
  // image_finemapping = fineMapping->FineMappingVertical(image_finemapping,6,0,-50,17,-PlateMargin,PlateMargin);
            image_finemapping = pr::fastdeskew(image_finemapping, 5);
            if(color_reverse_flag)
            {
                cv::bitwise_not(image_finemapping,image_finemapping);
                color_reverse_flag = 0;
            }
            //Segmentation-based

            if(SegmentationMethod==SEGMENTATION_BASED_METHOD)
            {
                image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 2, HorizontalPadding);
                cv::resize(image_finemapping, image_finemapping, cv::Size(136+HorizontalPadding, 36));
//            cv::imshow("image_finemapping",image_finemapping);
//            cv::waitKey(0);
                plateinfo.setPlateImage(image_finemapping);
                std::vector<cv::Rect> rects;

                plateSegmentation->segmentPlatePipline(plateinfo, 1, rects);
                plateSegmentation->ExtractRegions(plateinfo, rects);
                cv::copyMakeBorder(image_finemapping, image_finemapping, 0, 0, 0, 0, cv::BORDER_REPLICATE);
                plateinfo.setPlateImage(image_finemapping);
                generalRecognizer->SegmentBasedSequenceRecognition(plateinfo);
                plateinfo.decodePlateNormal(CH_PLATE_CODE);

            }

             //Segmentation-free
            else if(SegmentationMethod==SEGMENTATION_FREE_METHOD)
            {

                image_finemapping = fineMapping->FineMappingHorizon(image_finemapping,2, HorizontalPadding+0);

                cv::resize(image_finemapping, image_finemapping, cv::Size(136+HorizontalPadding, 36));
                plateinfo. setPlateImage(image_finemapping);
                std::pair<std::string,float> res = segmentationFreeRecognizer->SegmentationFreeForSinglePlate(plateinfo.getPlateImage(),CH_PLATE_CODE);
                plateinfo.confidence = res.second;
                plateinfo.setPlateName(res.first);
            }
            else{
                assert(SegmentationMethod>2);
            }
			plateinfo.Type = plateinfo.plateColor;
			if (plateinfo.confidence > 0.8)
			{

				if (plateinfo.getPlateName().find(CH_PLATE_CODE[68]) != -1)
				{
					plateinfo.Type = WHITE;
				}
				else if (plateinfo.getPlateName().find(CH_PLATE_CODE[67]) != -1)
				{
					plateinfo.Type = BLACK;
				}
				else if (plateinfo.getPlateName().find(CH_PLATE_CODE[65]) != -1 || plateinfo.getPlateName().find(CH_PLATE_CODE[69]) != -1)
				{
					plateinfo.Type = BLACK;
				}
				else if (plateinfo.getPlateName().size() == 9 && plateinfo.getPlateName().find(CH_PLATE_CODE[66]) == -1)
				{
					plateinfo.Type = GREEN;

				}
				else if (plateinfo.getPlateName().size() == 8 && plateinfo.Type == GREEN)
				{
					plateinfo.Type = BLUE;
				}
			}
			
            results.push_back(plateinfo);
        }

//        for (auto str:results) {
//            std::cout << str << std::endl;
//        }
        return results;

    }//namespace pr



}
