#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include "utility/utils.h"
#include "cnn_runtime/det2d/denet.h"

const static std::string model_path = "./TextNet.bin";
const static std::vector<std::string> input_name = {"data"};
const static std::vector<std::string> output_name = {"636", "662", "688"};
const char* class_name[4] = {"pear", "apple", "orange", "potato"};

void image_txt_infer(const std::string &image_dir, const std::string &image_txt_path){
    unsigned long time_start, time_end;
    std::vector<std::vector<float>> boxes;
    std::ofstream save_result;
    std::ifstream read_txt;
    std::string line_data;
    cv::Mat src_image;
    DeNet denet_process;
    if(denet_process.init(model_path, input_name, output_name) < 0)
    {
        std::cout << "DeNet init fail!" << std::endl;
        return;
    }

    read_txt.open(image_txt_path.data());
    if(!read_txt.is_open()){
        std::cout << image_txt_path << " not exits" << std::endl;
        return;
    }
    
    save_result.open("./det2d_result.txt");
    while(std::getline(read_txt, line_data)){
        boxes.clear();
        if(line_data.empty()){
            continue;
        }
        size_t index = line_data.find_first_of(' ', 0);
        std::string image_name = line_data.substr(0, index);
        std::stringstream image_path;
        image_path << image_dir << image_name;
        std::cout << image_path.str() << std::endl;
        src_image = cv::imread(image_path.str());
        time_start = get_current_time();
        boxes = denet_process.run(src_image);
        time_end = get_current_time();
        std::cout << "det2d cost time: " <<  (time_end - time_start)/1000.0  << "ms" << std::endl;

        save_result << image_name << "|";
        for (size_t i = 0; i < boxes.size(); ++i)
	    {
            float xmin = boxes[i][0];
            float ymin = boxes[i][1];
            float xmax = xmin + boxes[i][2];
            float ymax = ymin + boxes[i][3];
            int type = boxes[i][4];
            float confidence = boxes[i][5];
            save_result << class_name[type] << " " << confidence << " " << xmin 
                                << " " << ymin << " " << xmax << " " << ymax << "|";
	    }
        save_result << "\n";
    }
    read_txt.close();
    save_result.close();
}

int main()
{
    std::cout << "start..." << std::endl;
    const std::string image_dir = "./";
    const std::string image_txt_path = "./val.txt";
    image_txt_infer(image_dir, image_txt_path);
    std::cout << "End of game!!!" << std::endl;
    return 0;
}
