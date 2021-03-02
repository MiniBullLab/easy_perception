#ifndef NETPROCESS_H
#define NETPROCESS_H

#include <stdio.h>

#include "nnctrl.h"

#include "inference/common/common_log.h"
#include "inference/common/cavalry_process.h"

#define MAX_FILE_NAME_LEN			(256)

/**
 * @brief submodule to operate on nnctrl lib
 */
#define net_num 1

 struct net_match {
	uint8_t net_id;

	struct net_run_cfg net_rev;
	struct net_mem net_m;

    struct net_cfg cfg;
    struct net_input_cfg net_in;
    struct net_output_cfg net_out;
    struct net_result result;

    char net_file[MAX_FILE_NAME_LEN];
};

typedef struct nnctrl_ctx_s {
    uint8_t verbose;
    uint8_t reuse_mem;
    uint8_t cache_en;
    uint8_t buffer_id;
    uint8_t log_level;

    struct net_match net;
} nnctrl_ctx_t;

int init_net_context(nnctrl_ctx_t *nnctrl_ctx,
                     cavalry_ctx_t *cavalry_ctx,
                     uint8_t verbose, 
                     uint8_t cache_en);

void deinit_net_context(nnctrl_ctx_t *nnctrl_ctx, cavalry_ctx_t *cavalry_ctx);

int init_net(nnctrl_ctx_t *nnctrl_ctx, uint8_t verbose, uint8_t cache_en, uint8_t reuse_mem);

int load_net(nnctrl_ctx_t *nnctrl_ctx);

#endif //NETPROCESS_H