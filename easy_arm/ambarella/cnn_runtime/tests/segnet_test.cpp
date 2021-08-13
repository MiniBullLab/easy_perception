#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include "utility/utils.h"
#include "cnn_runtime/segment/segnet.h"

const static std::string model_path = "./segnet.bin";
const static std::string input_name = "0";
const static std::string output_name = "507";


void image_txt_infer(const std::string &image_dir, const std::string &image_txt_path){
    const std::string save_result_dir = "./seg_result/";
    unsigned long time_start, time_end;
    std::ifstream read_txt;
    std::string line_data;
    cv::Mat src_image;
    cv::Mat result_mat;
    SegNet seg_process;
    if(seg_process.init(model_path, input_name, output_name) < 0)
    {
        std::cout << "ClassNet init fail!" << std::endl;
        return;
    }

    read_txt.open(image_txt_path.data());
    if(!read_txt.is_open()){
        std::cout << image_txt_path << " not exits" << std::endl;
        return;
    }
    
    while(std::getline(read_txt, line_data)){
        if(line_data.empty()){
            continue;
        }
        size_t str_index = line_data.find_first_of(' ', 0);
        std::string image_name_post = line_data.substr(0, str_index);
        str_index = image_name_post.find_first_of('.', 0);
        std::string image_name = line_data.substr(0, str_index);
        std::stringstream save_path;
        std::stringstream image_path;
        image_path << image_dir << image_name_post;
        std::cout << image_path.str() << std::endl;
        src_image = cv::imread(image_path.str());
        time_start = get_current_time();
        seg_process.run(src_image);
        time_end = get_current_time();
        std::cout << "seg cost time: " <<  (time_end - time_start) / 1000.0  << "ms" << std::endl;

        save_path << save_result_dir << image_name << ".png";
        cv::imwrite(save_path.str(), result_mat);
    }
    read_txt.close();
}

int main()
{
    std::cout << "start..." << std::endl;
    const std::string image_dir = "";
    const std::string image_txt_path = "";
    image_txt_infer(image_dir, image_txt_path);
    std::cout << "End of game!!!" << std::endl;
    return 0;
}
