#ifndef NETPROCESS_H
#define NETPROCESS_H

#include "inference/common/common_log.h"
#include "inference/common/cavalry_process.h"

/**
 * @brief submodule to operate on nnctrl lib
 */
#define net_num (1)

int init_net_context(nnctrl_ctx_t *nnctrl_ctx,
                     cavalry_ctx_t *cavalry_ctx,
                     uint8_t verbose, 
                     uint8_t cache_en);

void deinit_net_context(nnctrl_ctx_t *nnctrl_ctx, cavalry_ctx_t *cavalry_ctx);

int init_net(nnctrl_ctx_t *nnctrl_ctx, uint8_t verbose, uint8_t cache_en, uint8_t reuse_mem);

int load_net(nnctrl_ctx_t *nnctrl_ctx);

// void set_net_io(nnctrl_ctx_t *nnctrl_ctx, const char* net_in_name, const char* net_out_name);

// int cnn_init(nnctrl_ctx_t *nnctrl_ctx, cavalry_ctx_t *cavalry_ctx, const char* model_path, \
//              const char* net_in_name, const char* net_out_name);

#endif //NETPROCESS_H