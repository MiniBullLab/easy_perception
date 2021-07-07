#include "inference/rec_text/textnet.h"
#include "inference/common/utils.h"
#include "inference/common/blob_define.h"
#include "inference/common/vproc_process.h"
#include "inference/common/net_process.h"
#include "inference/common/image_process.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const static int maxTextLength = 32;
const static int classNumber = 96;
const static int g_canvas_id = 1;
const static char characterSet[classNumber] = {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
                                               'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
                                               's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 
                                               'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                                               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
                                               'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', 
                                               '7', '8', '9', '0', '!', '"', '#', '$', '%', '&', 
                                               '\\', '\'', '(', ')', '*', '+', ',', '-', '.', '/', 
                                               ':', ';', '<', '=', '>', '?', '@', '[', ']', '^', 
                                               '_', '`', '{', '}', '~', '|', ' '};


static void set_net_io(nnctrl_ctx_t *nnctrl_ctx, const char* net_in_name, const char* net_out_name){
	nnctrl_ctx->net.net_in.in_num = 1;
    nnctrl_ctx->net.net_in.in_desc[0].name = net_in_name;
	nnctrl_ctx->net.net_in.in_desc[0].no_mem = 0;

	nnctrl_ctx->net.net_out.out_num = 1;
    nnctrl_ctx->net.net_out.out_desc[0].name = net_out_name; 
	nnctrl_ctx->net.net_out.out_desc[0].no_mem = 0; // let nnctrl lib allocate memory for output
}

static int init_param(nnctrl_ctx_t *nnctrl_ctx, const char* model_path)
{
    int rval = 0;
    memset(nnctrl_ctx, 0, sizeof(nnctrl_ctx_t));

    nnctrl_ctx->verbose = 0;
    nnctrl_ctx->reuse_mem = 1;
    nnctrl_ctx->cache_en = 1;
    nnctrl_ctx->buffer_id = g_canvas_id;
    nnctrl_ctx->log_level = 0;

    strcpy(nnctrl_ctx->net.net_file, model_path); 

    return rval;
}

static int textnet_run(TextNet::textnet_ctx_s *textnet_ctx, float *output)
{
    int rval = 0;
    nnctrl_ctx_t *nnctrl_ctx = &textnet_ctx->nnctrl_ctx;

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
    int output_p = nnctrl_ctx->net.net_out.out_desc[0].dim.pitch / 4;

    std::cout << "output size: " << "--output_c: " << output_c << "--output_h: " << output_h << "--output_w: " \
                                  << output_w << "--output_p: " << output_p << "--" << std::endl;

    for (int h = 0; h < output_h; h++)
    {
        memcpy(output + h * output_w, score_addr + h * output_w, output_w * sizeof(float));
    }

    return rval;
}

TextNet::TextNet()
{
    memset(&textnet_ctx, 0, sizeof(textnet_ctx_s));
    textnetOutput = NULL;
}

TextNet::~TextNet()
{
    deinit_net_context(&textnet_ctx.nnctrl_ctx, &textnet_ctx.cavalry_ctx);
    DPRINT_NOTICE("mtcnn_deinit\n");
    if(textnetOutput != NULL)
    {
        delete[] textnetOutput;
        textnetOutput = NULL;
    }
}

int TextNet::init(const std::string &modelPath, const std::string &inputName, const std::string &outputName)
{
    int rval = 0;
    init_param(&textnet_ctx.nnctrl_ctx, modelPath.c_str());
    nnctrl_ctx_t *nnctrl_ctx = &textnet_ctx.nnctrl_ctx; 

    rval = init_net_context(&textnet_ctx.nnctrl_ctx, &textnet_ctx.cavalry_ctx, 
                            nnctrl_ctx->verbose, nnctrl_ctx->cache_en);

    set_net_io(nnctrl_ctx, inputName.c_str(), outputName.c_str());
    rval = init_net(nnctrl_ctx, nnctrl_ctx->verbose, nnctrl_ctx->cache_en, nnctrl_ctx->reuse_mem);
    rval = load_net(nnctrl_ctx);

    if (rval < 0) {
        printf("init net context, return %d\n", rval);
    }

    textnetOutput = new float[maxTextLength * classNumber];

    return rval;
}

std::string TextNet::run(const cv::Mat &src_img)
{
    std::string result = "";
    float max_score = 0.0f;
    int pre_max_index = 0;
    int max_index = 0;
    preprocess(&textnet_ctx.nnctrl_ctx, src_img, 2);
    textnet_run(&textnet_ctx, textnetOutput);
    for (int row = 0; row < maxTextLength; row++) {
        max_score = 0.1f;
        max_index = 0;
        for (int col = 0; col < classNumber; col++) {
            if (textnetOutput[row * maxTextLength + col] > max_score) {
                max_score = textnetOutput[row * maxTextLength + col];
                max_index = col;
            }
        }
        if((max_index > 0) && !(row > 0 && max_index == pre_max_index))
        {
            std::cout << characterSet[max_index] << std::endl;
            result += characterSet[max_index];
        }
        pre_max_index = max_index;
    }
    return result;
}