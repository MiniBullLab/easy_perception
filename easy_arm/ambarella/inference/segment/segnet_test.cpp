/*******************************************************************************
 * segnet_test.c
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

#include "inference/common/utils.h"
#include "inference/common/vproc_process.h"
#include "inference/common/net_process.h"
#include "inference/common/image_process.h"

#define SEGMENT_NUM (200000)
#define SegOutputWidth (500)
#define SegOutputHeight (400)
#define SegOutputChannel (2)
#define THRESH (0.5f)

const static int g_canvas_id = 1;
const static char *net_in_name = "0";
const static char *net_out_name = "507";

typedef struct seg_ctx_s {
    cavalry_ctx_t cavalry_ctx;
    vproc_ctx_t vproc_ctx;
    nnctrl_ctx_t nnctrl_ctx;
} seg_ctx_t;

static void set_net_io(nnctrl_ctx_t *nnctrl_ctx){
	nnctrl_ctx->net.net_in.in_num = 1;
    nnctrl_ctx->net.net_in.in_desc[0].name = net_in_name;
	nnctrl_ctx->net.net_in.in_desc[0].no_mem = 0;

	nnctrl_ctx->net.net_out.out_num = 1;
    nnctrl_ctx->net.net_out.out_desc[0].name = net_out_name; 
	nnctrl_ctx->net.net_out.out_desc[0].no_mem = 0; // let nnctrl lib allocate memory for output
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

    strcpy(nnctrl_ctx->net.net_file, "./segnet.bin"); 

    return rval;
}

static int segment_init(seg_ctx_t *seg_ctx)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &seg_ctx->nnctrl_ctx; 

    rval = init_net_context(&seg_ctx->nnctrl_ctx, &seg_ctx->cavalry_ctx, 
                            nnctrl_ctx->verbose, nnctrl_ctx->cache_en);

    set_net_io(nnctrl_ctx);
    rval = init_net(nnctrl_ctx, nnctrl_ctx->verbose, nnctrl_ctx->cache_en, nnctrl_ctx->reuse_mem);
    rval = load_net(nnctrl_ctx);

    if (rval < 0) {
        printf("init net context, return %d\n", rval);
    }

    return rval;
}

static void segment_deinit(seg_ctx_t *seg_ctx)
{
    deinit_net_context(&seg_ctx->nnctrl_ctx, &seg_ctx->cavalry_ctx);
    DPRINT_NOTICE("segment_deinit\n");
}

int segment_run(seg_ctx_t *seg_ctx, float *output)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &seg_ctx->nnctrl_ctx;
    int output_pitch = get_output_pitch(nnctrl_ctx);

    rval = nnctrl_run_net(nnctrl_ctx->net.net_id, &nnctrl_ctx->net.result, NULL, NULL, NULL);

    if (rval < 0)
    {
        DPRINT_ERROR("nnctrl_run_net() failed, return %d\n", rval);
    }

    // parse the output of classnet
    if (nnctrl_ctx->cache_en) {
        cavalry_sync_cache(nnctrl_ctx->net.net_m.mem_size, nnctrl_ctx->net.net_m.phy_addr, 0, 1);
    }
    
     float *score_addr = (float *)(nnctrl_ctx->net.net_m.virt_addr
        + nnctrl_ctx->net.net_out.out_desc[0].addr - nnctrl_ctx->net.net_m.phy_addr);

    int output_c = nnctrl_ctx->net.net_out.out_desc[0].dim.depth;
    int output_h = nnctrl_ctx->net.net_out.out_desc[0].dim.height;
    int output_w = nnctrl_ctx->net.net_out.out_desc[0].dim.width;
    int output_p = nnctrl_ctx->net.net_out.out_desc[0].dim.pitch;

    std::cout << "output size: " << "--output_c: " << output_c << "--output_h: " << output_h << "--output_w: " \
                                  << output_w << "--output_p: " << output_p << "--" << std::endl;

    for (int h = 0; h < output_h; h++)
    {
        memcpy(output + h * output_w, score_addr + h * output_pitch, output_w * sizeof(float));
    }

    return rval;
}

void postprocess(const float* output, const int flag, const cv::Size src_size, 
                 const cv::Size net_input_size, cv::Mat &result_mat)
{
    cv::Mat seg_mat(net_input_size.height, net_input_size.width, CV_8UC3);
    uint8_t colorB[] = {0, 0};
    uint8_t colorG[] = {0, 0};
    uint8_t colorR[] = {0, 255};
    for (int row = 0; row < seg_mat.rows; row++) {
        for (int col = 0; col < seg_mat.cols; col++) {
            int posit;
            if (output[row * seg_mat.cols + col] > THRESH) {
                posit = 1;
            }
            else {
                posit = 0;
            }
            if(flag == 0){
                seg_mat.at<cv::Vec3b>(row, col) = cv::Vec3b(posit, posit, posit);
            }
            else if (flag == 1){
                seg_mat.at<cv::Vec3b>(row, col) = cv::Vec3b(colorB[posit], colorG[posit], colorR[posit]);
            }
        }
    }
    cv::resize(seg_mat, result_mat, src_size, 0, 0, cv::INTER_NEAREST);
}

void image_txt_infer(const std::string &image_dir, const std::string &image_txt_path){
    const std::string save_result_dir = "./seg_result/";
    unsigned long time_start, time_end;
    seg_ctx_t seg_ctx;
    std::vector<std::vector<float>> boxes;
    std::ifstream read_txt;
    std::string line_data;
    cv::Mat src_image;

    read_txt.open(image_txt_path.data());
    if(!read_txt.is_open()){
        std::cout << image_txt_path << " not exits" << std::endl;
        return;
    }
    
    memset(&seg_ctx, 0, sizeof(seg_ctx_t));
    init_param(&seg_ctx.nnctrl_ctx);
    segment_init(&seg_ctx);
    cv::Size net_input_size = get_input_size(&seg_ctx.nnctrl_ctx);
    float *seg_output = new float[net_input_size.height * net_input_size.width];
    cv::Mat result_mat;
    while(std::getline(read_txt, line_data)){
        boxes.clear();
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
        preprocess(&seg_ctx.nnctrl_ctx, src_image, 1);
        segment_run(&seg_ctx, seg_output);
        postprocess(seg_output, 0, cv::Size(src_image.cols, src_image.rows), net_input_size, result_mat);
        time_end = get_current_time();
        std::cout << "seg cost time: " <<  (time_end - time_start) / 1000.0  << "ms" << std::endl;

        save_path << save_result_dir << image_name << ".png";
        cv::imwrite(save_path.str(), result_mat);
    }
    delete[] seg_output;
    seg_output = NULL;
    read_txt.close();
    segment_deinit(&seg_ctx);
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
