
#include <signal.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/prctl.h>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "utility/utils.h"

#include "cnn_det3d/drivers/image_acquisition.h"
#include "cnn_det3d/drivers/tof_316_acquisition.h"

#include "cnn_det3d/network/tcp_process.h"
#include "cnn_det3d/network/network_process.h"

#include "cnn_runtime/det2d/denet.h"

#define TIME_MEASURE_LOOPS			(20)

#define CLASS_NUMBER (2)

#define FRAME_HEADER1	(0x55)
#define FRAME_HEADER2	(0xAA)
#define YUV_DATA_LENGTH	(8 + 4 + 4 + 4 + 4 + IMAGE_YUV_SIZE)
#define TOTAL_YUV_SIZE	(2 + 1 + 4 + YUV_DATA_LENGTH)
#define TOF_DATA_LENGTH (8 + 4 + 4 + 4 + 4 + TOF_SIZE)
#define TOTAL_TOF_SIZE  (2 + 1 + 4 + TOF_DATA_LENGTH)

volatile int run_flag = 1;
volatile int run_send = 0;
volatile int run_denet = 0;

static TCPProcess tcp_process;
static TOF316Acquisition tof_geter;
static ImageAcquisition image_geter;
static NetWorkProcess network_process;

void static fill_data(unsigned char* addr, int data)
{
	addr[0] = (data >> 24) & 0xFF;
	addr[1] = (data >> 16) & 0xFF;
	addr[2] = (data >>  8) & 0xFF;
	addr[3] = (data >>  0) & 0xFF;
}

static void *send_tof_pthread(void *thread_params)
{
	uint64_t debug_time = 0;
	unsigned char send_data[TOTAL_TOF_SIZE];
	int length = 0;
	long time_stamp = 0;
	int data_type = 0, width = 0, height = 0;

	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;

	prctl(PR_SET_NAME, "send_tof_pthread");
	tof_geter.set_up();
	while(run_send > 0)
	{
		tof_geter.get_tof_Z(&send_data[31]);
		send_data[0] = FRAME_HEADER1;
		send_data[1] = FRAME_HEADER2;
		send_data[2] = 0x07;
		length = TOF_DATA_LENGTH;
		fill_data(&send_data[3], length);

		time_stamp = get_time_stamp();
		fill_data(&send_data[7], (time_stamp >> 32) & 0xFFFFFFFF);
		fill_data(&send_data[11], time_stamp & 0xFFFFFFFF);

		data_type = 1;
		fill_data(&send_data[15], data_type);

		width = DEPTH_WIDTH;
		fill_data(&send_data[19], width);

		height = DEPTH_HEIGTH;
		fill_data(&send_data[23], height);

		tcp_process.send_data(send_data, TOTAL_TOF_SIZE);
	}
	pthread_detach(pthread_self());
	LOG(WARNING) << "send_tof_pthread quit";
	return NULL;
}

static void *send_yuv_pthread(void *thread_params)
{
	uint64_t debug_time = 0;
	unsigned char send_data[TOTAL_YUV_SIZE];
	int length = 0;
	long time_stamp = 0;
	int data_type = 0, width = 0, height = 0;

	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;

	prctl(PR_SET_NAME, "send_yuv_pthread");
	while(run_send > 0)
	{
 		image_geter.get_yuv(&send_data[31]);
		send_data[0] = FRAME_HEADER1;
		send_data[1] = FRAME_HEADER2;
		send_data[2] = 0x06;
		length = YUV_DATA_LENGTH;
		fill_data(&send_data[3], length);

		time_stamp = get_time_stamp();
		fill_data(&send_data[7], (time_stamp >> 32) & 0xFFFFFFFF);
		fill_data(&send_data[11], time_stamp & 0xFFFFFFFF);

		data_type = 0;
		fill_data(&send_data[15], data_type);

		width = IMAGE_WIDTH;
		fill_data(&send_data[19], width);

		height = IMAGE_HEIGHT;
		fill_data(&send_data[23], height);

		tcp_process.send_data(send_data, TOTAL_YUV_SIZE);
	}
	pthread_detach(pthread_self());
	LOG(WARNING) << "send_yuv_pthread quit";
	return NULL;
}

static void *send_data_pthread(void *thread_params)
{
	int rval = 0;
	pthread_t tof_pthread_id = 0;
	pthread_t image_pthread_id = 0;
	while(run_flag > 0)
	{
		while (run_send > 0)
		{
			if(tcp_process.accept_connect() >= 0)
			{
				rval = pthread_create(&tof_pthread_id, NULL, send_tof_pthread, NULL);
				if(rval >= 0)
				{
					rval = pthread_create(&image_pthread_id, NULL, send_yuv_pthread, NULL);
					if(rval < 0)
					{
						run_send = 0;
						LOG(ERROR) << "create pthread fail!";
					}
				}
				else
				{
					run_send = 0;
					LOG(ERROR) << "create pthread fail!";
				}
				if (tof_pthread_id > 0) {
					pthread_join(tof_pthread_id, NULL);
				}
				if (image_pthread_id > 0) {
					pthread_join(image_pthread_id, NULL);
				}
			}
		}
		usleep(100000);
	}
	pthread_detach(pthread_self());
	LOG(WARNING) << "send_data_pthread quit";
	return NULL;
}

static void *run_denet_pthread(void *thread_params)
{
	const std::string model_path = "/data/denet.bin";
	const std::vector<std::string> input_name = {"data"};
	const std::vector<std::string> output_name = {"det_output0", "det_output1", "det_output2"};
	const char* class_name[CLASS_NUMBER] = {"green_strawberry", "strawberry"};

	std::vector<std::vector<float>> boxes;
	std::vector<TOF316Acquisition::PointCloud> boxesCloud;

	cv::Mat src_image;
	cv::Mat depth_map = cv::Mat::zeros(cv::Size(DEPTH_WIDTH, DEPTH_HEIGTH), CV_32FC1);
	TOF316Acquisition::PointCloud src_cloud;
	
	DeNet denet_process;
	// Time measurement
	uint64_t save_time = 0;
	uint64_t start_time = 0;
	uint64_t debug_time = 0;
	float sum_time = 0.0f;
	uint32_t loop_count = 1;

	cv::Mat M= (cv::Mat_<float>(3,3) << 1055.3f, 0, 635.4f, 0, 1052.0f, 394.7f, 0, 0, 1);
	cv::Mat rvec = (cv::Mat_<float>(3,1) << 0.026032152475774608f,
					-0.0200025179954166f,
					-3.130275855466499f);
    cv::Mat tvec = (cv::Mat_<float>(3,1) << -0.02651449597600853f,
						-0.0116211406889809f,
						0.02123382406509555f);
	cv::Mat camera_rotation;
	cv::Rodrigues(rvec, camera_rotation);
	cv::Mat RT(3, 4, camera_rotation.type()); // T is 4x4
	RT(cv::Range(0,3), cv::Range(0,3)) = camera_rotation * 1; // copies R into T
	RT(cv::Range(0,3), cv::Range(3,4)) = tvec * 1; // copies tvec into T

	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;

	prctl(PR_SET_NAME, "run_denet_pthread");

	if(denet_process.init(model_path, input_name, output_name, CLASS_NUMBER, 0.5f) < 0)
    {
		LOG(ERROR) << "DeNet init fail!";
        return NULL;
    }

	while(run_flag > 0)
	{
		if(run_send == 0)
		{
			tof_geter.get_tof_pc(src_cloud);
			// tof_geter.get_tof_Z(depth_map);
			image_geter.get_image(src_image);
		}
		else
		{
			usleep(50000);
		}
		save_time = gettimeus();
		while(run_denet > 0)
		{
			start_time = gettimeus();
			tof_geter.get_tof_pc(src_cloud);
			// tof_geter.get_tof_Z(depth_map);
			image_geter.get_image(src_image);
			if(src_image.empty())
			{
				LOG(ERROR) << "DeNet get image fail!";
				continue;
			}
			boxes = denet_process.run(src_image);
			LOG(WARNING) << "car count:" << boxes.size();

			if(boxes.size() > 0)
			{
				boxesCloud.clear();
				boxesCloud.resize(boxes.size());
				for (size_t i = 0; i < src_cloud.size(); i++)
				{
					cv::Mat point = (cv::Mat_<float>(4, 1) << src_cloud[i].x, src_cloud[i].y, src_cloud[i].z, 1);
					cv::Mat uv(3, 1, CV_32F);
					uv = M * RT * point;
					if (uv.at<float>(2) <= 0.00001f)
					{
						LOG(ERROR) << "uv.at<float>(2)=0 !";
						break;
					}
					float u = uv.at<float>(0) / uv.at<float>(2);
					float v = uv.at<float>(1) / uv.at<float>(2);
					int px = int(u + 0.5);
					int py = int(v + 0.5);
					if (0 <= px && px < src_image.cols && 0 <= py && py < src_image.rows)
					{
						for (size_t index = 0; index < boxes.size(); index++)
						{
							int xmin = (int)boxes[index][0];
							int ymin = (int)boxes[index][1];
							int xmax = (int)(xmin + boxes[index][2]);
							int ymax = (int)(ymin + boxes[index][3]);
							if (px > xmin && px < xmax && py > ymin && py < ymax)
							{
								boxesCloud[index].push_back(src_cloud[i]);
								src_image.at<cv::Vec3b>(py, px)[0] = 0;
								src_image.at<cv::Vec3b>(py, px)[1] = 0;
								src_image.at<cv::Vec3b>(py, px)[2] = 255;
							}
						}
					}
				}

				std::stringstream result;
				for (size_t index = 0; index < boxes.size(); ++index)
				{
					int type = boxes[index][4];
					float sum_x = 0;
					float sum_y = 0;
					float sum_z = 0;
					for (size_t i = 0; i < boxesCloud[index].size(); i++)
					{
						sum_x += boxesCloud[index][i].x;
						sum_y += boxesCloud[index][i].y;
						sum_z += boxesCloud[index][i].z;
					}
					if(boxesCloud[index].size() > 0)
					{
						sum_x /= boxesCloud[index].size();
						sum_y /= boxesCloud[index].size();
						sum_z /= boxesCloud[index].size();
						result << index << " " << sum_x << " " << sum_y << " " << sum_z << "|";
					}
				}
				if(result.str() != "")
					network_process.send_result(result.str(), 50);
			}

			for (size_t i = 0; i < boxes.size(); ++i)
			{
				float xmin = boxes[i][0];
				float ymin = boxes[i][1];
				float xmax = xmin + boxes[i][2];
				float ymax = ymin + boxes[i][3];
				int type = boxes[i][4];
				float confidence = boxes[i][5];
				cv::rectangle(src_image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 255), 2, 4);
			}

			std::stringstream filename_image;
			filename_image << "img_" << save_time << ".png";
        	cv::imwrite(filename_image.str(), src_image);

			// std::stringstream filename_tof;
			// filename_tof << "tof_" << save_time << ".txt";
			// tof_geter.dump_z_txt(filename_tof.str(), depth_map);

			// std::stringstream filename_tof;
			// filename_tof << "tof_" << save_time << ".bin";
			// tof_geter.dump_bin(filename_tof.str(), src_cloud);

			// std::stringstream filename_tof1;
			// filename_tof1 << "tof_" << save_time << ".ply";
			// tof_geter.dump_ply(filename_tof1.str().c_str(), src_cloud);

			sum_time += (gettimeus() - start_time);
			++loop_count;
			if (loop_count == TIME_MEASURE_LOOPS) {
				LOG(WARNING) << "det average time [per " << TIME_MEASURE_LOOPS << " loops]:" << sum_time / (1000 * TIME_MEASURE_LOOPS) << "ms";
				sum_time = 0;
				loop_count = 1;
			}
		}
	}
	network_process.send_error(1);
	LOG(WARNING) << "run_denet_pthread quit！";
	return NULL;
}

static int start_all()
{
	int rval = 0;
	pthread_t denet_pthread_id = 0;
	pthread_t send_pthread_id = 0;
	if(network_process.start() < 0)
	{
		rval = -1;
		run_flag = 0;
		LOG(ERROR) << "start network fail!";
		return rval;
	}
	else
	{
		LOG(INFO) << "start network success";
	}
	if(tof_geter.start() < 0)
	{
		rval = -1;
		run_flag = 0;
		LOG(ERROR) << "start tof fail!";
	}
	LOG(INFO) << "start tof success";
	if(image_geter.start() < 0)
	{
		rval = -1;
		run_flag = 0;
		LOG(ERROR) << "start image fail!";
	}
	LOG(INFO) << "start image success";
	rval = pthread_create(&denet_pthread_id, NULL, run_denet_pthread, NULL);
	if(rval < 0)
	{
		rval = -1;
		run_flag = 0;
		LOG(ERROR) << "start denet fail!";
	}
	LOG(INFO) << "start denet success";
	rval = pthread_create(&send_pthread_id, NULL, send_data_pthread, NULL);
	if(rval < 0)
	{
		rval = -1;
		run_flag = 0;
		LOG(ERROR) << "start send fail!";
	}
	LOG(INFO) << "start send success";
	while(run_flag > 0)
	{
		int result = network_process.process_recv();
		if(result == 200)
		{
			run_denet = 0;
			run_flag = 0;
		}
		else if(result == 300)
		{
			run_denet = 1;
		}
		else if(result == 400)
		{
			run_denet = 0;
		}
		else if(result == 500)
		{
			run_send = 1;
		}
		else if(result == 600)
		{
			run_send = 0;
		}
	}
	if (denet_pthread_id > 0) {
		pthread_join(denet_pthread_id, NULL);
		denet_pthread_id = 0;
	}
	LOG(WARNING) << "denet pthread release";
	if (send_pthread_id > 0) {
		pthread_join(send_pthread_id, NULL);
		send_pthread_id = 0;
	}
	LOG(WARNING) << "send pthread release";
	network_process.stop();
	LOG(WARNING) << "Main thread quit";
	return rval;
}

static void sigstop(int signal_number)
{
	run_flag = 0;
	run_denet = 0;
	network_process.send_post();
	LOG(WARNING) << "sigstop msg, exit";
}

static void SignalHandle(const char* data, int size) {
    std::string str = data;
	network_process.send_error(19);
	run_flag = 0;
	run_denet = 0;
	network_process.send_post();
    LOG(FATAL) << str;
}

int main(int argc, char **argv)
{
	int rval = 0;
	google::InitGoogleLogging(argv[0]);

	FLAGS_log_dir = "/data/glog_file";
	// google::SetLogDestination(google::GLOG_ERROR, "/data/glogfile/logerror");
	google::InstallFailureSignalHandler();
	google::InstallFailureWriter(&SignalHandle); 

	FLAGS_stderrthreshold = 1;
	FLAGS_colorlogtostderr = true; 
	FLAGS_logbufsecs = 5;    //缓存的最大时长，超时会写入文件
	FLAGS_max_log_size = 10; //单个日志文件最大，单位M
	FLAGS_logtostderr = false; //设置为true，就不会写日志文件了
	// FLAGS_alsologtostderr = true;
	FLAGS_minloglevel = 0;
	FLAGS_stop_logging_if_full_disk = true;
 
	signal(SIGINT, sigstop);
	signal(SIGQUIT, sigstop);
	signal(SIGTERM, sigstop);

	if(tof_geter.open_tof() == 0 && image_geter.open_camera() == 0)
	{
		if(tcp_process.socket_init() >= 0 && network_process.init_network() >= 0)
		{
			if(start_all() < 0)
			{
				LOG(ERROR) << "start fail!";
			}
			tof_geter.stop();
			image_geter.stop();
			LOG(WARNING) << "Main thread quit";
		}
		else
		{
			LOG(ERROR) << "start_all fail!";
		}
	}
	LOG(INFO) << "All Quit";
	google::ShutdownGoogleLogging();
	return rval;
}


