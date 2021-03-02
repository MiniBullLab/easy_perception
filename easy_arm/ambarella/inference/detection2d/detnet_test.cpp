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

#include "inference/common/utils.h"
#include "inference/common/vproc_process.h"
#include "inference/common/net_process.h"
#include "inference/common/image_process.h"

#include "inference/detection2d/yolov3.h"

#define OUTPUT_NUM (3)

const static int g_canvas_id = 1;
const char* net_in_name = "data";
const char* net_out_name_1 = "636";
const char* net_out_name_2 = "662";
const char* net_out_name_3 = "688";
const char* class_name[4] = {"pear", "apple", "orange", "potato"};

typedef struct det2d_ctx_s {
    cavalry_ctx_t cavalry_ctx;
    vproc_ctx_t vproc_ctx;
    nnctrl_ctx_t nnctrl_ctx;
} det2d_ctx_t;

static void set_net_io(nnctrl_ctx_t *nnctrl_ctx)
{
    nnctrl_ctx->net.net_in.in_num = 1;
	nnctrl_ctx->net.net_in.in_desc[0].name = net_in_name;
	nnctrl_ctx->net.net_in.in_desc[0].no_mem = 0;

	nnctrl_ctx->net.net_out.out_num = 3;
	nnctrl_ctx->net.net_out.out_desc[0].name = net_out_name_1;
	nnctrl_ctx->net.net_out.out_desc[0].no_mem = 0; // let nnctrl lib allocate memory for output
    nnctrl_ctx->net.net_out.out_desc[1].name = net_out_name_2;
	nnctrl_ctx->net.net_out.out_desc[1].no_mem = 0; // let nnctrl lib allocate memory for output
    nnctrl_ctx->net.net_out.out_desc[2].name = net_out_name_3;
	nnctrl_ctx->net.net_out.out_desc[2].no_mem = 0; // let nnctrl lib allocate memory for output
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

    strcpy(nnctrl_ctx->net.net_file, "./detnet.bin"); 

    return rval;
}

static int det2d_init(det2d_ctx_t *det2d_ctx)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &det2d_ctx->nnctrl_ctx; 

    rval = init_net_context(nnctrl_ctx, &det2d_ctx->cavalry_ctx, 
                            nnctrl_ctx->verbose, nnctrl_ctx->cache_en);

    set_net_io(nnctrl_ctx);
    rval = init_net(nnctrl_ctx, nnctrl_ctx->verbose, nnctrl_ctx->cache_en, nnctrl_ctx->reuse_mem);
    rval = load_net(nnctrl_ctx);

    if (rval < 0) {
        printf("init net context, return %d\n", rval);
    }

    return rval;
}

static void det2d_deinit(det2d_ctx_t *det2d_ctx)
{
    deinit_net_context(&det2d_ctx->nnctrl_ctx, &det2d_ctx->cavalry_ctx);
    DPRINT_NOTICE("det2d_deinit\n");
}

int det2d_run(det2d_ctx_t *det2d_ctx, float *output[OUTPUT_NUM])
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &det2d_ctx->nnctrl_ctx;

    rval = nnctrl_run_net(nnctrl_ctx->net.net_id, &nnctrl_ctx->net.result, NULL, NULL, NULL);

    if (rval < 0)
    {
        DPRINT_ERROR("nnctrl_run_net() failed, return %d\n", rval);
    }

    // parse the output of classnet
    if (nnctrl_ctx->cache_en) {
        cavalry_sync_cache(nnctrl_ctx->net.net_m.mem_size, nnctrl_ctx->net.net_m.phy_addr, 0, 1);
    }
    
    for (int i = 0; i < OUTPUT_NUM; i++)
	{
        float *score_addr = (float *)(nnctrl_ctx->net.net_m.virt_addr
                            + nnctrl_ctx->net.net_out.out_desc[i].addr - nnctrl_ctx->net.net_m.phy_addr);


        int output_c = nnctrl_ctx->net.net_out.out_desc[i].dim.depth;
        int output_h = nnctrl_ctx->net.net_out.out_desc[i].dim.height;
        int output_w = nnctrl_ctx->net.net_out.out_desc[i].dim.width;
        int output_p = nnctrl_ctx->net.net_out.out_desc[i].dim.pitch;

		output[i] = score_addr;	
    }

    return rval;
}

std::vector<std::vector<float>> postprocess(const cv::Size src_size, const cv::Size dst_size, float *output[OUTPUT_NUM])
{
    float ratio;
    cv::Size pad_size;

    std::vector<std::vector<float>> final_results;
	final_results = yolo_run(output[0], output[1], output[2]);

    get_square_size(src_size, dst_size, ratio, pad_size);

    for (size_t i = 0; i < final_results.size(); ++i)
    {
        final_results[i][0] = (final_results[i][0] - floor((float)pad_size.width / 2)) / ratio;
        final_results[i][1] = (final_results[i][1] - floor((float)pad_size.height / 2)) / ratio;
        final_results[i][2] = final_results[i][2] / ratio;
        final_results[i][3] = final_results[i][3] / ratio;

        if (final_results[i][0] < 1.0) {
            final_results[i][0] = 1.0;
        }
        if (final_results[i][1] < 1.0) {
            final_results[i][1] = 1.0;
        }
        if (final_results[i][0] + final_results[i][2] >= src_size.width) {
            final_results[i][2] = floor((src_size.width - final_results[i][0]));
        }
        if (final_results[i][1] + final_results[i][3] >= src_size.height) {
            final_results[i][3] = floor((src_size.height - final_results[i][1]));
        }
    }
    // std::cout << "final results size: " << final_results.size() << std::endl;

    return final_results;
}

void image_txt_infer(const std::string &image_dir, const std::string &image_txt_path){
    unsigned long time_start, time_end;
    det2d_ctx_t det2d_ctx;
    std::vector<std::vector<float>> boxes;
    std::ofstream save_result;
    std::ifstream read_txt;
    std::string line_data;
    cv::Mat src_image;

    read_txt.open(image_txt_path.data());
    if(!read_txt.is_open()){
        std::cout << image_txt_path << " not exits" << std::endl;
        return;
    }
    
    memset(&det2d_ctx, 0, sizeof(det2d_ctx_t));
    init_param(&det2d_ctx.nnctrl_ctx);
    det2d_init(&det2d_ctx);
    cv::Size net_input_size = get_input_size(&det2d_ctx.nnctrl_ctx);
    save_result.open("./det2d_result.txt");
    while(std::getline(read_txt, line_data)){
        float *output[OUTPUT_NUM];
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
        preprocess(&det2d_ctx.nnctrl_ctx, src_image, 1);
        det2d_run(&det2d_ctx, output);
        boxes = postprocess(cv::Size(src_image.cols, src_image.rows), net_input_size, output);
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
    det2d_deinit(&det2d_ctx);
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
