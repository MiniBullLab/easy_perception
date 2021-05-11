/*******************************************************************************
 * detnet_test.c
 *
 * Author: foweiw
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <dirent.h>

#include <sstream>

#include<opencv2/highgui.hpp>


#include "inference/common/utils.h"
#include "inference/common/vproc_process.h"
#include "inference/common/net_process.h"
#include "inference/common/image_process.h"

#include "inference/posenet/posenet_postprocess.h"

#define OUTPUT_NUM (1)

const static int g_canvas_id = 1;
const static char* net_in_name = "data";
const static char* net_out_name_1 = "concat_stage7";
//const static char* net_out_name_2 = "365";

const static int nPoints = 18;

const static std::vector<std::pair<int,int>> posePairs = {
    {1,2}, {1,5}, {2,3}, {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10}, {1,11}, {11,12}, {12,13},
    {1,0}, {0,14}, {14,16}, {0,15}, {15,17}, {2,17},
    {5,16}
};

typedef struct posenet_ctx_s {
    cavalry_ctx_t cavalry_ctx;
    vproc_ctx_t vproc_ctx;
    nnctrl_ctx_t nnctrl_ctx;
} posenet_ctx_t;

static void set_net_io(nnctrl_ctx_t *nnctrl_ctx)
{
    nnctrl_ctx->net.net_in.in_num = 1;
	nnctrl_ctx->net.net_in.in_desc[0].name = net_in_name;
	nnctrl_ctx->net.net_in.in_desc[0].no_mem = 0;

	nnctrl_ctx->net.net_out.out_num = OUTPUT_NUM;
	nnctrl_ctx->net.net_out.out_desc[0].name = net_out_name_1;
	nnctrl_ctx->net.net_out.out_desc[0].no_mem = 0; // let nnctrl lib allocate memory for output
    // nnctrl_ctx->net.net_out.out_desc[1].name = net_out_name_2;
	// nnctrl_ctx->net.net_out.out_desc[1].no_mem = 0; // let nnctrl lib allocate memory for output
}

static int init_param(nnctrl_ctx_t *nnctrl_ctx)
{
    int rval = 0;
    memset(nnctrl_ctx, 0, sizeof(nnctrl_ctx_t));

    nnctrl_ctx->verbose = 0;
    nnctrl_ctx->reuse_mem = 1;
    nnctrl_ctx->cache_en = 1;
    nnctrl_ctx->buffer_id = g_canvas_id;
    nnctrl_ctx->log_level = 0;

    strcpy(nnctrl_ctx->net.net_file, "./posenet.bin"); 

    return rval;
}

static int posenet_init(posenet_ctx_t *posenet_ctx)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &posenet_ctx->nnctrl_ctx; 

    rval = init_net_context(nnctrl_ctx, &posenet_ctx->cavalry_ctx, 
                            nnctrl_ctx->verbose, nnctrl_ctx->cache_en);

    set_net_io(nnctrl_ctx);
    rval = init_net(nnctrl_ctx, nnctrl_ctx->verbose, nnctrl_ctx->cache_en, nnctrl_ctx->reuse_mem);
    rval = load_net(nnctrl_ctx);

    if (rval < 0) {
        printf("init net context, return %d\n", rval);
    }

    return rval;
}

static void posenet_deinit(posenet_ctx_t *posenet_ctx)
{
    deinit_net_context(&posenet_ctx->nnctrl_ctx, &posenet_ctx->cavalry_ctx);
    DPRINT_NOTICE("posenet_deinit\n");
}

int posenet_run(posenet_ctx_t *posenet_ctx, const cv::Size src_size, std::vector<cv::Mat>& netOutputParts)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &posenet_ctx->nnctrl_ctx;

    rval = nnctrl_run_net(nnctrl_ctx->net.net_id, &nnctrl_ctx->net.result, NULL, NULL, NULL);

    if (rval < 0)
    {
        DPRINT_ERROR("nnctrl_run_net() failed, return %d\n", rval);
    }

    // parse the output of classnet
    if (nnctrl_ctx->cache_en) {
        cavalry_sync_cache(nnctrl_ctx->net.net_m.mem_size, nnctrl_ctx->net.net_m.phy_addr, 0, 1);
    }

    netOutputParts.clear();

    float *score_addr = (float *)(nnctrl_ctx->net.net_m.virt_addr
        + nnctrl_ctx->net.net_out.out_desc[0].addr - nnctrl_ctx->net.net_m.phy_addr);


    int output_c = nnctrl_ctx->net.net_out.out_desc[0].dim.depth;
    int output_h = nnctrl_ctx->net.net_out.out_desc[0].dim.height;
    int output_w = nnctrl_ctx->net.net_out.out_desc[0].dim.width;
    int output_p = nnctrl_ctx->net.net_out.out_desc[0].dim.pitch;
    int size = nnctrl_ctx->net.net_out.out_desc[0].size;

    std::cout << "output size: " << "--output_c: " << output_c << "--output_h: " << output_h << "--output_w: " \
                                  << output_w << "--output_p: " << output_p << "--" << size << std::endl;

    std::ofstream save_result;

    netOutputParts.clear();
    for(int c = 0; c < output_c; ++c)
    {
        cv::Mat resizedPart;
        cv::Mat part(output_h, output_w, CV_32FC1);
        for (int h = 0; h < output_h; h++)
        {
            memcpy(part.data + h * output_w, score_addr + c * h * output_p, output_w * sizeof(float));
        }
        cv::resize(part, resizedPart, src_size);
        netOutputParts.push_back(resizedPart);
    }

    save_result.open("temp.bin", std::ofstream::binary);
    save_result.write(reinterpret_cast<const char*>(score_addr), sizeof(float)*size);
    save_result.close();

    // std::ofstream save_result;

    // for (int i = OUTPUT_NUM-1; i >= 0; i--)
	// {
    //     float *score_addr = (float *)(nnctrl_ctx->net.net_m.virt_addr
    //                         + nnctrl_ctx->net.net_out.out_desc[i].addr - nnctrl_ctx->net.net_m.phy_addr);


    //     int output_c = nnctrl_ctx->net.net_out.out_desc[i].dim.depth;
    //     int output_h = nnctrl_ctx->net.net_out.out_desc[i].dim.height;
    //     int output_w = nnctrl_ctx->net.net_out.out_desc[i].dim.width;
    //     int output_p = nnctrl_ctx->net.net_out.out_desc[i].dim.pitch;
    //     int size = nnctrl_ctx->net.net_out.out_desc[i].size;

    //     std::cout << "output size: " << "--output_c: " << output_c << "--output_h: " << output_h << "--output_w: " \
    //                               << output_w << "--output_p: " << output_p << "--" << size << std::endl;

	// 	// output[i] = score_addr;	
    //     std::stringstream save_path;
    //     save_path << i << "_output.bin";
    //     save_result.open(save_path.str(), std::ofstream::binary);
    //     save_result.write(reinterpret_cast<const char*>(score_addr), sizeof(float)*size);
    //     save_result.close();

    //     for(int c = 0; c < output_c;++c)
    //     {
    //         cv::Mat resizedPart;
    //         cv::Mat part(output_h, output_w, CV_32FC1, cv::Scalar(0));
    //         for (int h = 0; h < output_h; h++)
    //         {
    //             memcpy(part.data + h * output_w, score_addr + c * h * output_p, output_w * sizeof(float));
    //         }
    //         cv::resize(part, resizedPart, src_size);
    //         netOutputParts.push_back(resizedPart);
    //     }

    // }

    return rval;
}

std::vector<std::vector<KeyPoint>> postprocess(const cv::Size src_size, const cv::Size dst_size, const std::vector<cv::Mat>& netOutputParts)
{
    //std::vector<cv::Mat> netOutputParts;
    std::vector<std::vector<KeyPoint>> final_results;
    //splitNetOutputBlobToParts(output[1], output[0], src_size, netOutputParts);
    getPostnetResult(netOutputParts, final_results);
    return final_results;
}

void image_txt_infer(const std::string &image_dir, const std::string &image_txt_path)
{
    unsigned long frame_index = 0;
    unsigned long time_start, time_end;
    posenet_ctx_t pose_ctx;
    std::vector<std::vector<KeyPoint>> result;
    std::vector<cv::Mat> netOutputParts;
    std::ifstream read_txt;
    std::string line_data;
    cv::Mat src_image;

    read_txt.open(image_txt_path.data());
    if(!read_txt.is_open()){
        std::cout << image_txt_path << " not exits" << std::endl;
        return;
    }
    
    memset(&pose_ctx, 0, sizeof(posenet_ctx_t));
    init_param(&pose_ctx.nnctrl_ctx);
    posenet_init(&pose_ctx);
    cv::Size net_input_size = get_input_size(&pose_ctx.nnctrl_ctx);

    while(std::getline(read_txt, line_data)){
        result.clear();
        netOutputParts.clear();
        if(line_data.empty()){
            continue;
        }
        size_t index = line_data.find_first_of(' ', 0);
        std::string image_name = line_data.substr(0, index);
        std::stringstream image_path;
        image_path << image_dir << image_name;
        std::cout << frame_index << image_path.str() << std::endl;
        src_image = cv::imread(image_path.str());
        time_start = get_current_time();
        preprocess(&pose_ctx.nnctrl_ctx, src_image, 0);
        posenet_run(&pose_ctx, cv::Size(src_image.cols, src_image.rows), netOutputParts);
        time_end = get_current_time();
        std::cout << "posenet cost time: " <<  (time_end - time_start)/1000.0  << "ms" << std::endl;
        getPostnetResult(netOutputParts, result);
        //result = postprocess(cv::Size(src_image.cols, src_image.rows), net_input_size, output);
        time_end = get_current_time();
        std::cout << "posenet cost time: " <<  (time_end - time_start)/1000.0  << "ms" << std::endl;

        for(int i = 0; i< nPoints-1;++i){
            for(int n  = 0; n < result.size();++n){
                const std::pair<int,int>& posePair = posePairs[i];
                const KeyPoint& kpA = result[n][posePair.first];
                const KeyPoint& kpB = result[n][posePair.second];
                if(kpA.probability < 0 || kpB.probability < 0){
                    continue;
                }
                cv::line(src_image, kpA.point, kpB.point, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
            }
        }
        std::stringstream save_path;
        save_path << frame_index << "_li.png";
        cv::imwrite(save_path.str(), src_image);
        frame_index++;
    }
    read_txt.close();
    posenet_deinit(&pose_ctx);
}

int main()
{
    std::cout << "start..." << std::endl;
    const std::string image_dir = "./pose_img/";
    const std::string image_txt_path = "img.txt";
    image_txt_infer(image_dir, image_txt_path);
    std::cout << "End of game!!!" << std::endl;
    return 0;
}
