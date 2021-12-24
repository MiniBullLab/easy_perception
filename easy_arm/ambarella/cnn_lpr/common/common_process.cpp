#include "common_process.h"
#include <iostream>

#define DEFAULT_STATE_BUF_NUM		(5)
#if defined(OLD_CODE)
#define DEFAULT_STREAM_ID			(2)
#define DEFAULT_CHANNEL_ID			(2)
#else
#define DEFAULT_STREAM_ID			(2)
#define DEFAULT_CHANNEL_ID			(3)
#endif
#define DEFAULT_RGB_TYPE			(1) /* 0: RGB, 1:BGR */

int init_param(global_control_param_t *G_param)
{
	int rval = 0;
	LOG(INFO) << "init ...";
	memset(G_param, 0, sizeof(global_control_param_t));

	G_param->channel_id = DEFAULT_CHANNEL_ID;
	G_param->stream_id = DEFAULT_STREAM_ID;

	G_param->state_buf_num = DEFAULT_STATE_BUF_NUM;
	G_param->overlay_x_offset = DEFAULT_X_OFFSET;
	G_param->overlay_highlight_sec = DEFAULT_HIGHLIGHT_SEC;
	G_param->overlay_clear_sec = DEFAULT_CLEAR_SEC;
	G_param->overlay_text_width_ratio = DEFAULT_WIDTH_RATIO;
	G_param->abort_if_preempted = 1;
	G_param->debug_en = 0;
	G_param->verbose = 0;

	LOG(INFO) << "init sucess";

	return rval;
}

int env_init(global_control_param_t *G_param)
{
	int rval = 0;
	LOG(INFO) << "env_init";
	do {
		RVAL_OK(ea_env_open(EA_ENV_ENABLE_IAV
			| EA_ENV_ENABLE_CAVALRY
			| EA_ENV_ENABLE_VPROC
			| EA_ENV_ENABLE_NNCTRL
			| EA_ENV_ENABLE_OSD_STREAM));
#if defined(OLD_CODE)
		G_param->img_resource = ea_img_resource_new(EA_PYRAMID,
					(void *)(unsigned long)G_param->channel_id);
#else
		G_param->img_resource = ea_img_resource_new(EA_CANVAS,
			(void *)(unsigned long)G_param->channel_id);
#endif
		RVAL_ASSERT(G_param->img_resource != NULL);
#ifdef IS_SHOW
		RVAL_OK(init_overlay_tool(G_param->stream_id,
			G_param->overlay_x_offset, G_param->overlay_highlight_sec,
			G_param->overlay_clear_sec, G_param->overlay_text_width_ratio,
			DEFAULT_OVERLAY_LICENSE_NUM, G_param->debug_en));
#endif
		RVAL_OK(init_state_buffer_param(&G_param->ssd_result_buf,
			G_param->state_buf_num, (uint16_t)sizeof(ea_img_resource_data_t),
			30, (G_param->debug_en >= INFO_LEVEL)));
		pthread_mutex_init(&G_param->access_buffer_mutex, NULL);
		if (G_param->abort_if_preempted) {
			pthread_mutex_init(&G_param->vp_access_lock, NULL);
		}
		sem_init(&G_param->sem_readable_buf, 0, 0);
		RVAL_OK(ea_set_preprocess_priority_on_current_process(VPROC_PRIORITY, EA_VP));
	} while(0);
	LOG(INFO) << "env_init success";
	return rval;
}

void env_deinit(global_control_param_t *G_param)
{
	pthread_mutex_destroy(&G_param->access_buffer_mutex);
	if (G_param->abort_if_preempted) {
		pthread_mutex_destroy(&G_param->vp_access_lock);
	}
	sem_destroy(&G_param->sem_readable_buf);
	deinit_state_buffer_param(&G_param->ssd_result_buf);
#ifdef IS_SHOW
	deinit_overlay_tool();
#endif
	ea_img_resource_free(G_param->img_resource);
	G_param->img_resource = NULL;
	ea_env_close();
}

int tensor2mat_yuv2bgr_nv12(ea_tensor_t *tensor, cv::Mat &bgr)
{
	int rval = EA_SUCCESS;
	cv::Mat nv12(ea_tensor_shape(tensor)[2] +
		(ea_tensor_related(tensor) == NULL ? 0 : ea_tensor_shape(ea_tensor_related(tensor))[2]),
		ea_tensor_shape(tensor)[3], CV_8UC1);
	uint8_t *p_src = NULL;
	uint8_t *p_dst = NULL;
	size_t h = 0;

	do {
		RVAL_ASSERT(ea_tensor_shape(tensor)[1] == 1);

		p_src = (uint8_t *)ea_tensor_data_for_read(tensor, EA_CPU);
		p_dst = nv12.data;
		for (h = 0; h < ea_tensor_shape(tensor)[2]; h++) {
			memcpy(p_dst, p_src, ea_tensor_shape(tensor)[3]);
			p_src += ea_tensor_pitch(tensor);
			p_dst += ea_tensor_shape(tensor)[3];
		}

		if (ea_tensor_related(tensor)) {
			p_src = (uint8_t *)ea_tensor_data_for_read(ea_tensor_related(tensor), EA_CPU);
			for (h = 0; h < ea_tensor_shape(ea_tensor_related(tensor))[2]; h++) {
				memcpy(p_dst, p_src, ea_tensor_shape(ea_tensor_related(tensor))[3]);
				p_src += ea_tensor_pitch(ea_tensor_related(tensor));
				p_dst += ea_tensor_shape(ea_tensor_related(tensor))[3];
			}
		}

		#if CV_VERSION_MAJOR < 4
			cv::cvtColor(nv12, bgr, CV_YUV2BGR_NV12);
		#else
			cv::cvtColor(nv12, bgr, COLOR_YUV2BGR_NV12);
		#endif
	} while (0);

	return rval;
}

int tensor2mat_yuv2bgr_nv12(ea_tensor_t *tensor, const uint16_t pitch, cv::Mat &bgr)
{
	int rval = EA_SUCCESS;
	size_t img_shape[4] = {1, 3, bgr.rows, bgr.cols};
	ea_tensor_t *img_tensor = ea_tensor_new(EA_U8, img_shape, pitch);
	std::vector<cv::Mat> channel_s;
	uint8_t *p_src = NULL;
	uint8_t *p_dst = NULL;
	size_t h = 0;
	cv::split(bgr, channel_s);
	do {
		RVAL_ASSERT(ea_tensor_shape(tensor)[1] == 1);
		RVAL_OK(ea_cvt_color_resize(tensor, img_tensor, EA_COLOR_YUV2BGR_NV12, EA_VP));
		
		p_src = (uint8_t *)ea_tensor_data_for_read(img_tensor, EA_CPU);
		p_dst = channel_s[0].data;
		for (h = 0; h < ea_tensor_shape(img_tensor)[2]; h++) {
			memcpy(p_dst, p_src, ea_tensor_shape(img_tensor)[3]);
			p_src += ea_tensor_pitch(img_tensor);
			p_dst += ea_tensor_shape(img_tensor)[3];
		}

		p_dst = channel_s[1].data;
		for (h = 0; h < ea_tensor_shape(img_tensor)[2]; h++) {
			memcpy(p_dst, p_src, ea_tensor_shape(img_tensor)[3]);
			p_src += ea_tensor_pitch(img_tensor);
			p_dst += ea_tensor_shape(img_tensor)[3];
		}

		p_dst = channel_s[2].data;
		for (h = 0; h < ea_tensor_shape(img_tensor)[2]; h++) {
			memcpy(p_dst, p_src, ea_tensor_shape(img_tensor)[3]);
			p_src += ea_tensor_pitch(img_tensor);
			p_dst += ea_tensor_shape(img_tensor)[3];
		}
		cv::merge(channel_s, bgr);
	} while (0);
	ea_tensor_free(img_tensor);
	img_tensor = NULL;
	return rval;
}

void swapYUV_I420toNV12(const unsigned char* i420bytes, unsigned char* nv12bytes, int width, int height)
{
    int nLenY = width * height;
    int nLenU = nLenY / 4;

    memcpy(nv12bytes, i420bytes, width * height);

    for (int i = 0; i < nLenU; i++)
    {
        nv12bytes[nLenY + 2 * i] = i420bytes[nLenY + i];                    // U
        nv12bytes[nLenY + 2 * i + 1] = i420bytes[nLenY + nLenU + i];        // V
    }
}

int mat2tensor_yuv_nv12(cv::Mat &yuv_i420, ea_tensor_t *tensor)
{
	int rval = EA_SUCCESS;
	int width = ea_tensor_shape(tensor)[3];
	int height = ea_tensor_shape(tensor)[2];
	cv::Mat nv12 = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
	uint8_t *p_src = NULL;
	uint8_t *p_dst = NULL;
	size_t h;
	swapYUV_I420toNV12(yuv_i420.data, nv12.data, yuv_i420.cols, yuv_i420.rows * 2 / 3);
	// std::cout << "nv12:" << yuv_i420.cols << " " << yuv_i420.rows << " " << ea_tensor_shape(tensor)[2] << std::endl;
	p_src = nv12.data;
	p_dst = (uint8_t *)ea_tensor_data_for_write(tensor, EA_CPU);
	for (h = 0; h < height; h++) {
		memcpy(p_dst, p_src, width);
		p_src += width;
		p_dst += ea_tensor_pitch(tensor);
	}

	// if (ea_tensor_related(tensor)) {
	// 	p_dst = (uint8_t *)ea_tensor_data_for_write(ea_tensor_related(tensor), EA_CPU);
	// 	for (h = 0; h < ea_tensor_shape(ea_tensor_related(tensor))[2]; h++) {
	// 		memcpy(p_dst, p_src, ea_tensor_shape(ea_tensor_related(tensor))[3]);
	// 		p_src += ea_tensor_shape(ea_tensor_related(tensor))[3];
	// 		p_dst += ea_tensor_pitch(ea_tensor_related(tensor));
	// 	}
	// }
	return rval;
}

int create_yuv_nv12_tensor(const unsigned char* addr, const int width, const int height, ea_tensor_t *tensor)
{
	int rval = EA_SUCCESS;
	int tensor_height = ea_tensor_shape(tensor)[2];
	cv::Mat nv12 = cv::Mat(tensor_height, width, CV_8UC1, cv::Scalar(0));
	uint8_t *p_src = NULL;
	uint8_t *p_dst = NULL;
	size_t h;
	std::cout << "111111111111111111111" << std::endl;
	swapYUV_I420toNV12(addr, nv12.data, width, height);
	std::cout << "222222222222222222222" << std::endl;
	p_src = nv12.data;
	p_dst = (uint8_t *)ea_tensor_data_for_write(tensor, EA_CPU);
	std::cout << "3333333333333333333333333" << std::endl;
	for (h = 0; h < tensor_height; h++) {
		memcpy(p_dst, p_src, width);
		p_src += width;
		p_dst += ea_tensor_pitch(tensor);
	}
	std::cout << "4444444444444444444444" << std::endl;
	return rval;
}

void fill_data(unsigned char* addr, int data)
{
	addr[0] = (data >> 24) & 0xFF;
	addr[1] = (data >> 16) & 0xFF;
	addr[2] = (data >>  8) & 0xFF;
	addr[3] = (data >>  0) & 0xFF;
}