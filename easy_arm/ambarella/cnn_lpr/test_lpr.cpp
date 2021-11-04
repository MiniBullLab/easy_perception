
#include <signal.h>
#include <stdint.h>

#include "cnn_lpr/common/common_process.h"
#include "cnn_lpr/ssd/det_process.h"
#include "cnn_lpr/lpr/lpr_process.h"

#include "cnn_lpr/tof/tof_acquisition.h"
#include "cnn_lpr/tof/tof_data_process.h"
#include "cnn_lpr/tof/vibebgs.h"

#include "cnn_lpr/net_process/net_process.h"

#include "utility/utils.h"

#define TIME_MEASURE_LOOPS			(100)

#define DEPTH_WIDTH (240)
#define DEPTH_HEIGTH (180)

//#define IS_SHOW 

static float lpr_confidence = 0;
static bbox_param_t lpr_bbox;
static int width_diff = 0;
static std::string lpr_result = "";

volatile int has_lpr = 0;
static pthread_mutex_t result_mutex;
static pthread_mutex_t ssd_mutex;

volatile int run_flag = 1;
volatile int run_lpr = 0;

static TOFAcquisition tof_geter;
static NetProcess net_process;

static void *run_lpr_pthread(void *lpr_param_thread)
{
	int rval;
	ssd_lpr_thread_params_t *lpr_param =
		(ssd_lpr_thread_params_t*)lpr_param_thread;
	global_control_param_t *G_param = lpr_param->G_param;
	LPR_ctx_t LPR_ctx;

	// Detection result param
	bbox_param_t bbox_param[MAX_DETECTED_LICENSE_NUM];
	draw_plate_list_t draw_plate_list;
	uint16_t license_num = 0;
	license_list_t license_result;
	state_buffer_t *ssd_mid_buf;
	ea_img_resource_data_t * data = NULL;
	ea_tensor_t *img_tensor = NULL;

	// Time mesurement
	uint64_t start_time = 0;
	uint64_t debug_time = 0;
	float sum_time = 0.0f;
	float average_license_num = 0.0f;
	uint32_t loop_count = 1;
	uint32_t debug_en = G_param->debug_en;

	bbox_param_t pre_lpr_bbox = {0};

	do {
		memset(&LPR_ctx, 0, sizeof(LPR_ctx));
		memset(&draw_plate_list, 0, sizeof(draw_plate_list));
		memset(&bbox_param, 0, sizeof(bbox_param));

		LPR_ctx.img_h = lpr_param->height;
		LPR_ctx.img_w = lpr_param->width;
		RVAL_OK(init_LPR(&LPR_ctx, G_param));
		RVAL_OK(alloc_single_state_buffer(&G_param->ssd_result_buf, &ssd_mid_buf));

		while (run_flag) {
			while(run_lpr > 0)
			{
				RVAL_OK(lpr_critical_resource(&license_num, bbox_param,
				ssd_mid_buf, G_param));
				start_time = gettimeus();
				data = (ea_img_resource_data_t *)ssd_mid_buf->img_resource_addr;
				if (license_num == 0) {
					RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, data));
					continue;
				}
				img_tensor = data->tensor_group[G_param->lpr_pyd_idx];
				RVAL_OK(LPR_run(&LPR_ctx, img_tensor, license_num,
					(void*)bbox_param, &license_result));
#ifdef IS_SHOW
				draw_overlay_preprocess(&draw_plate_list, &license_result,
					bbox_param, G_param);
#endif
				if(license_result.license_num > 0)
				{
					pthread_mutex_lock(&result_mutex);
					float width1 = bbox_param[0].norm_max_x - bbox_param[0].norm_min_x;
					float width2 = pre_lpr_bbox.norm_max_x - pre_lpr_bbox.norm_min_x;
					width_diff = static_cast<int>(abs(width1 - width2));
					lpr_bbox.norm_min_x = bbox_param[0].norm_min_x;
					lpr_bbox.norm_min_y = bbox_param[0].norm_min_y;
					lpr_bbox.norm_max_x = bbox_param[0].norm_max_x;
					lpr_bbox.norm_max_y = bbox_param[0].norm_max_y;
					pre_lpr_bbox = bbox_param[0];
					LOG(INFO) << "LPR:"  << license_result.license_info[0].text << " " << license_result.license_info[0].conf;
					if (license_result.license_info[0].conf > G_param->recg_threshold && \
						strlen(license_result.license_info[0].text) == CHINESE_LICENSE_STR_LEN && \
						license_result.license_info[0].conf > lpr_confidence)
						{
							lpr_result = license_result.license_info[0].text;
							lpr_confidence = license_result.license_info[0].conf;
							LOG(INFO) << "LPR:"  << lpr_result << " " << lpr_confidence;
						}
					pthread_mutex_unlock(&result_mutex);
				}
				TIME_MEASURE_START(debug_en);
				RVAL_OK(set_overlay_image(img_tensor, &draw_plate_list));
				TIME_MEASURE_END("[LPR] LPR draw overlay time", debug_en);
				RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, data));
				sum_time += (gettimeus() - start_time);
				++loop_count;
				average_license_num += license_num;
				if (loop_count == TIME_MEASURE_LOOPS) {
					float average_time1 = sum_time / (1000 * TIME_MEASURE_LOOPS);
					float average_time2 = (average_license_num > 0.0f) ? (sum_time / (1000 * average_license_num)) : 0.0f;
					LOG(INFO) << "[" << TIME_MEASURE_LOOPS  << "loops] LPR average time license_num " << " " << average_license_num / TIME_MEASURE_LOOPS;
					LOG(INFO) << "average time:"<< average_time1 << "per license cost time:" << average_time2;
					sum_time = 0;
					loop_count = 1;
					average_license_num = license_num;
				}
				if (debug_en == DEBUG_LEVEL) {
					run_flag = 0;
					LOG(INFO) << "In debug mode, stop after one loop!";
				}
			}
			usleep(20000);
		}
	} while (0);
	do {
		run_flag = 0;
		free_single_state_buffer(ssd_mid_buf);
		LPR_deinit(&LPR_ctx);
		LOG(INFO) << "LPR thread quit.";
	} while (0);

	return NULL;
}


static void *run_ssd_pthread(void *ssd_thread_params)
{
	int rval = 0;
	unsigned long long int frame_number = 0;
	uint32_t i = 0;
	ssd_lpr_thread_params_t *ssd_param =
		(ssd_lpr_thread_params_t*)ssd_thread_params;
	global_control_param_t *G_param = ssd_param->G_param;
	// SSD param
	SSD_ctx_t SSD_ctx;
	int ssd_result_num = 0;
	bbox_param_t scaled_license_plate;
	state_buffer_t *ssd_mid_buf;
	ssd_net_final_result_t ssd_net_result;
	bbox_list_t bbox_list;

	// image related
	ea_tensor_t *img_tensor = NULL;
	ea_img_resource_data_t data;
	uint32_t dsp_pts = 0;

	// Time measurement
	uint64_t start_time = 0;
	uint64_t debug_time = 0;
	float sum_time = 0.0f;
	uint32_t loop_count = 1;
	uint32_t debug_en = G_param->debug_en;

    // cv::Mat bgr(ssd_param->height * 2 / 3, ssd_param->width, CV_8UC3);

	do {
		memset(&ssd_net_result, 0, sizeof(ssd_net_result));
		memset(&data, 0, sizeof(data));
		memset(&SSD_ctx, 0, sizeof(SSD_ctx_t));
		memset(&scaled_license_plate, 0, sizeof(scaled_license_plate));

		RVAL_OK(init_ssd(&SSD_ctx, G_param, ssd_param->height, ssd_param->width));
		ssd_net_result.dproc_ssd_result = (dproc_ssd_detection_output_result_t *)
			malloc(SSD_ctx.vp_result_info.max_dproc_ssd_result_num *
			sizeof(dproc_ssd_detection_output_result_t));
		RVAL_ASSERT(ssd_net_result.dproc_ssd_result != NULL);
		RVAL_OK(alloc_single_state_buffer(&G_param->ssd_result_buf, &ssd_mid_buf));

		while (run_flag) {
			while(run_lpr > 0){
				RVAL_OK(ea_img_resource_hold_data(G_param->img_resource, &data));
				RVAL_ASSERT(data.tensor_group != NULL);
				RVAL_ASSERT(data.tensor_num >= 1);
				img_tensor = data.tensor_group[G_param->ssd_pyd_idx];
				dsp_pts = data.dsp_pts;
				// SAVE_TENSOR_IN_DEBUG_MODE("SSD_pyd.jpg", img_tensor, debug_en);
				if(frame_number % 80 == 0)
				{
					// std::stringstream filename;
					// filename << "image_" << frame_number << ".jpg";
					// std::cout << "tensor channel:" << ea_tensor_shape(img_tensor)[1] << std::endl;
					// RVAL_OK(ea_tensor_to_jpeg(img_tensor, EA_TENSOR_COLOR_MODE_YUV_NV12, filename.str().c_str()));
					has_lpr = 0;
				}
				frame_number++;

				start_time = gettimeus();

				// TIME_MEASURE_START(debug_en);
				// RVAL_OK(tensor2mat_yuv2bgr_nv12(img_tensor, bgr));
				// TIME_MEASURE_END("[SSD] yuv to bgr time", debug_en);

				TIME_MEASURE_START(debug_en);
				RVAL_OK(ea_cvt_color_resize(img_tensor, SSD_ctx.net_input.tensor,
					EA_COLOR_YUV2BGR_NV12, EA_VP));
				TIME_MEASURE_END("[SSD] preprocess time", debug_en);

				TIME_MEASURE_START(debug_en);
				RVAL_OK(ssd_net_run_vp_forward(&SSD_ctx.ssd_net_ctx));
				ea_tensor_sync_cache(SSD_ctx.ssd_net_ctx.output_loc_tensor, EA_VP, EA_CPU);
				ea_tensor_sync_cache(SSD_ctx.ssd_net_ctx.output_conf_tensor, EA_VP, EA_CPU);
				TIME_MEASURE_END("[SSD] network time", debug_en);

				TIME_MEASURE_START(debug_en);
				ssd_net_result.ssd_det_num = 0;
				memset(&ssd_net_result.labels[0][0], 0,
					SSD_NET_MAX_LABEL_NUM * SSD_NET_MAX_LABEL_LEN);
				memset(ssd_net_result.dproc_ssd_result, 0,
					SSD_ctx.vp_result_info.max_dproc_ssd_result_num *
					sizeof(dproc_ssd_detection_output_result_t));
				RVAL_OK(ssd_net_run_arm_nms(&SSD_ctx.ssd_net_ctx,
					SSD_ctx.vp_result_info.loc_dram_addr,
					SSD_ctx.vp_result_info.conf_dram_addr, &ssd_net_result));
				TIME_MEASURE_END("[SSD] ARM NMS time", debug_en);

				TIME_MEASURE_START(debug_en);
				ssd_result_num = min(ssd_net_result.ssd_det_num, MAX_DETECTED_LICENSE_NUM);
				bbox_list.bbox_num = min(ssd_result_num, MAX_OVERLAY_PLATE_NUM);
				ssd_critical_resource(ssd_net_result.dproc_ssd_result, &data,
					bbox_list.bbox_num, ssd_mid_buf, G_param);

				for (i = 0; i < bbox_list.bbox_num; ++i) {
					upscale_normalized_rectangle(ssd_net_result.dproc_ssd_result[i].bbox.x_min,
					ssd_net_result.dproc_ssd_result[i].bbox.y_min,
					ssd_net_result.dproc_ssd_result[i].bbox.x_max,
					ssd_net_result.dproc_ssd_result[i].bbox.y_max,
					DRAW_LICNESE_UPSCALE_W, DRAW_LICNESE_UPSCALE_H, &scaled_license_plate);
					bbox_list.bbox[i].norm_min_x = scaled_license_plate.norm_min_x;
					bbox_list.bbox[i].norm_min_y = scaled_license_plate.norm_min_y;
					bbox_list.bbox[i].norm_max_x = scaled_license_plate.norm_max_x;
					bbox_list.bbox[i].norm_max_y = scaled_license_plate.norm_max_y;

					has_lpr = 1;
				}

				LOG(INFO) << "lpr box count:" << bbox_list.bbox_num;

				RVAL_OK(set_overlay_bbox(&bbox_list));
				RVAL_OK(show_overlay(dsp_pts));
				TIME_MEASURE_END("[SSD] post-process time", debug_en);

				sum_time += (gettimeus() - start_time);
				++loop_count;
				if (loop_count == TIME_MEASURE_LOOPS) {
					LOG(INFO) << "SSD average time [per " << TIME_MEASURE_LOOPS << " loops]:" << sum_time / (1000 * TIME_MEASURE_LOOPS) << "ms";
					sum_time = 0;
					loop_count = 1;
				}
			}
			has_lpr = 0;
			usleep(20000);
#ifdef IS_SHOW
			bbox_list.bbox_num = 0;
			RVAL_OK(set_overlay_bbox(&bbox_list));
		    RVAL_OK(show_overlay(dsp_pts));
#endif
		}
	} while (0);
	do {
		run_flag = 0;
		if (ssd_net_result.dproc_ssd_result != NULL) {
			free(ssd_net_result.dproc_ssd_result);
		}
		ssd_net_deinit(&SSD_ctx.ssd_net_ctx);
		free_single_state_buffer(ssd_mid_buf);
		LOG(INFO) << "SSD thread quit.";
	} while (0);

	return NULL;
}

// static void offline_point_cloud_process()
// {
// 	std::string cloud_dir = "./point_cloud/";
// 	std::vector<std::string> cloud_files;
// 	int point_count = 0;
// 	unsigned long long int frame_number = 0;
// 	std::vector<int> point_cout_list;
// 	TOFAcquisition::PointCloud src_cloud;
// 	TOFAcquisition::PointCloud bg_cloud;
// 	point_cout_list.clear();
// 	read_bin(bg_point_cloud_file, bg_cloud);
// 	ListImages(cloud_dir, cloud_files);
//     std::cout << "total Test cloud : " << cloud_files.size() << std::endl;
// 	while(run_flag > 0)
// 	{
// 		for (size_t index = 0; index < cloud_files.size(); index++) {
// 			std::stringstream temp_str;
//         	temp_str << cloud_dir << cloud_files[index];
// 			std::cout << temp_str.str() << std::endl;
// 			read_bin(temp_str.str(), src_cloud);
// 			if(src_cloud.size() > 10)
// 			{
// 				run_lpr = 1;
// 				if(frame_number % 10 == 0)
// 				{
// 					vote_in_out(point_cout_list);
// 					point_cout_list.clear();
// 				}
// 				point_count = compute_point_count(bg_cloud, src_cloud);
// 				std::cout << "point count:" << point_count << std::endl;
// 				if(point_count > 10)
// 					point_cout_list.push_back(point_count);
// 			}
// 		}
// 	}
// }

static void point_cloud_process(const global_control_param_t *G_param)
{
	uint64_t debug_time = 0;
	uint32_t debug_en = G_param->debug_en;
	int send_count = 0;
	int bg_point_count = 0;
	// int is_in = -1;
	unsigned long long int process_number = 0;
	unsigned long long int no_process_number = 0;
	cv::Mat filter_map;
	cv::Mat bg_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH),CV_8UC1);
	cv::Mat depth_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH),CV_8UC1);
	std::vector<int> result_list;
	std::vector<int> point_cout_list;
	TOFAcquisition::PointCloud src_cloud;

	cv::Mat img_bgmodel;
	cv::Mat img_output;
	IBGS *bgs = new ViBeBGS();

	result_list.clear();
	point_cout_list.clear();

	tof_geter.get_tof_data(src_cloud, depth_map);
	// cv::medianBlur(depth_map, bg_map, 3);
	cv::GaussianBlur(depth_map, bg_map, cv::Size(9, 9), 3.5, 3.5);
	cv::imwrite("./bg.png", bg_map);

	while(run_flag > 0)
	{
		TIME_MEASURE_START(debug_en);
		tof_geter.get_tof_data(src_cloud, depth_map);
		TIME_MEASURE_END("[point_cloud] get TOF cost time", debug_en);

		TIME_MEASURE_START(debug_en);
		cv::GaussianBlur(depth_map, filter_map, cv::Size(9, 9), 3.5, 3.5);
		TIME_MEASURE_END("[point_cloud] filtering cost time", debug_en);

		TIME_MEASURE_START(debug_en);
		bgs->process(filter_map, img_output, img_bgmodel);
		bg_point_count = static_cast<int>(cv::sum(img_output / 255)[0]);
		LOG(INFO) << "bg_point_count:" << bg_point_count;
		TIME_MEASURE_END("[point_cloud] bgs cost time", debug_en);

		// if(process_number % 1 == 0)
		// {
		// 	std::stringstream filename;
		// 	filename << "point_cloud" << process_number << ".png";
		// 	cv::imwrite(filename.str(), img_output);
		// 	// dump_bin(filename.str(), src_cloud);
		// }
		// process_number++;

		if(bg_point_count > 50)
		{
			tof_geter.set_up();
			run_lpr = 1;
			if(has_lpr == 1)
			{
				point_cout_list.push_back(bg_point_count);
				// process_number++;
				// if(process_number % 10 == 0)
				// {
				// 	is_in = vote_in_out(point_cout_list);
				// 	result_list.push_back(is_in);
				// 	point_cout_list.clear();
				// }
			}
		}
		if(bg_point_count <= 50 || has_lpr == 0)
		{
			no_process_number++;
			if(no_process_number % 10 == 0)
			{
				int final_result = vote_in_out(point_cout_list);
				//int final_result = get_in_out(result_list);
				int point_count = compute_depth_map(bg_map, filter_map);
				LOG(INFO) << "final point_count:" << point_count << " " << final_result;
				if(final_result == 0 && point_count >= 500)
				{
					final_result = 0;
				}
				else if(final_result == 0 && point_count < 100)
				{
					final_result = 1;
				}
				else if(final_result == 1 && point_count >= 500)
				{
					final_result = 0;
				}
				LOG(INFO) << "final_result:" << final_result;
				pthread_mutex_lock(&result_mutex);
				LOG(INFO) << "width_diff:" << width_diff;
				if(send_count == 1 && final_result == 0 && width_diff < 10)
				{
					final_result = -1;
				}
				if(final_result >= 0)
				{
					if(lpr_result != "" && lpr_confidence > 0)
					{
						net_process.send_result(lpr_result, final_result);
						send_count = 1;
						lpr_confidence = 0;
					}
					else if(final_result == 1 && lpr_result != "")
					{
						net_process.send_result(lpr_result, final_result);
						lpr_result = "";
						lpr_confidence = 0;
					}
					if(final_result == 1)
					{
						send_count = 0;
					}
				}
				pthread_mutex_unlock(&result_mutex);
				point_cout_list.clear();
				result_list.clear();
				process_number = 0;
				no_process_number = 0;
				has_lpr = 0;
				run_lpr = 0;
				tof_geter.set_sleep();
			}
		}
		else
		{
			no_process_number = 0;
		}
		TIME_MEASURE_END("[point_cloud] cost time", debug_en);
	}
	delete bgs;
    bgs = NULL;
	LOG(INFO) << "stop point cloud process";
}

static int start_all_lpr(global_control_param_t *G_param)
{
	int rval = 0;
	pthread_t ssd_pthread_id = 0;
	pthread_t lpr_pthread_id = 0;
	ssd_lpr_thread_params_t lpr_thread_params;
	ssd_lpr_thread_params_t ssd_thread_params;

	ea_tensor_t *img_tensor = NULL;
	ea_img_resource_data_t data;

	if(tof_geter.start() < 0)
	{
		rval = -1;
		run_flag = 0;
	}
	LOG(INFO) << "start tof success";
	
	if(net_process.init_network() < 0)
	{
		rval = -1;
		run_flag = 0;
	}
	LOG(INFO) << "net init success";

	do {
		pthread_mutex_init(&result_mutex, NULL);
		pthread_mutex_init(&ssd_mutex, NULL);

		memset(&lpr_thread_params, 0 , sizeof(lpr_thread_params));
		memset(&data, 0, sizeof(data));
		RVAL_OK(ea_img_resource_hold_data(G_param->img_resource, &data));
		RVAL_ASSERT(data.tensor_group != NULL);
		RVAL_ASSERT(data.tensor_num >= 1);
		img_tensor = data.tensor_group[G_param->lpr_pyd_idx];
		lpr_thread_params.height = ea_tensor_shape(img_tensor)[2];
		lpr_thread_params.width = ea_tensor_shape(img_tensor)[3];
		lpr_thread_params.pitch = ea_tensor_pitch(img_tensor);
		lpr_thread_params.G_param = G_param;
		img_tensor = data.tensor_group[G_param->ssd_pyd_idx];
		ssd_thread_params.height = ea_tensor_shape(img_tensor)[2];
		ssd_thread_params.width = ea_tensor_shape(img_tensor)[3];
		ssd_thread_params.pitch = ea_tensor_pitch(img_tensor);
		ssd_thread_params.G_param = G_param;
		RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &data));
		rval = pthread_create(&ssd_pthread_id, NULL, run_ssd_pthread, (void*)&ssd_thread_params);
		RVAL_ASSERT(rval == 0);
		rval = pthread_create(&lpr_pthread_id, NULL, run_lpr_pthread, (void*)&lpr_thread_params);
		RVAL_ASSERT(rval == 0);
	} while (0);
	LOG(INFO) << "start_ssd_lpr success";

	point_cloud_process(G_param);

	if (lpr_pthread_id > 0) {
		pthread_join(lpr_pthread_id, NULL);
	}
	if (ssd_pthread_id > 0) {
		pthread_join(ssd_pthread_id, NULL);
	}
	pthread_mutex_destroy(&result_mutex);
	pthread_mutex_destroy(&ssd_mutex);
	LOG(INFO) << "Main thread quit";
	return rval;
}

static void sigstop(int signal_number)
{
	run_lpr = 0;
	run_flag = 0;
	tof_geter.stop();
	LOG(INFO) << "sigstop msg, exit";
	return;
}

void SignalHandle(const char* data, int size) {
    std::string str = data;
	run_lpr = 0;
	run_flag = 0;
	tof_geter.stop();
    LOG(ERROR) << str;
}

int main(int argc, char **argv)
{
	int rval = 0;
	global_control_param_t G_param;

	google::InitGoogleLogging(argv[0]);

	google::SetLogDestination(google::INFO, "/data/glogfile/loginfo");   
	google::SetLogDestination(google::WARNING, "/data/glogfile/logwarn");   
	google::SetLogDestination(google::GLOG_ERROR, "/data/glogfile/logerror");
	google::InstallFailureSignalHandler();
	google::InstallFailureWriter(&SignalHandle); 

	FLAGS_colorlogtostderr = true; 
	FLAGS_logbufsecs = 20;    //缓存的最大时长，超时会写入文件
	FLAGS_max_log_size = 10; //单个日志文件最大，单位M
	FLAGS_logtostderr = false; //设置为true，就不会写日志文件了
	// FLAGS_alsologtostderr = true;
	FLAGS_minloglevel = 0;
	FLAGS_stderrthreshold = 1;
	FLAGS_stop_logging_if_full_disk = true;

	signal(SIGINT, sigstop);
	signal(SIGQUIT, sigstop);
	signal(SIGTERM, sigstop);
	if(tof_geter.open_tof() == 0)
	{
		do {
			RVAL_OK(init_param(&G_param));
			RVAL_OK(env_init(&G_param));
			RVAL_OK(start_all_lpr(&G_param));
		}while(0);
		env_deinit(&G_param);
	}
	LOG(INFO) << "All Quit";
	google::ShutdownGoogleLogging();
	return rval;
}


