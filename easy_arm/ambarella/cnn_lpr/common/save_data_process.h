#ifndef _SAVE_DATA_PROCESS_H_
#define _SAVE_DATA_PROCESS_H_

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "cnn_lpr/drivers/image_acquisition.h"
#include "cnn_lpr/drivers/tof_316_acquisition.h"

//#define IS_SAVE

#if defined(ONLY_SAVE_DATA) || defined(ONLY_SEND_DATA)
#define SAVE_TOF_BUFFER_SIZE (1)
#define SAVE_IMAGE_BUFFER_SIZE (1)
#elif defined(OFFLINE_DATA) || defined(IS_SAVE)
#define SAVE_TOF_BUFFER_SIZE (5)
#define SAVE_IMAGE_BUFFER_SIZE (5)
#else
#define SAVE_TOF_BUFFER_SIZE (1)
#define SAVE_IMAGE_BUFFER_SIZE (1)
#endif

struct SaveTofBuffer  
{  	
    cv::Mat buffer[SAVE_TOF_BUFFER_SIZE];
    long buffer_stamp[SAVE_TOF_BUFFER_SIZE];
    pthread_mutex_t lock; /* 互斥体lock 用于对缓冲区的互斥操作 */  
    int readpos, writepos; /* 读写指针*/  
    pthread_cond_t notempty; /* 缓冲区非空的条件变量 */  
    pthread_cond_t notfull; /* 缓冲区未满的条件变量 */  
};

struct SaveImageBuffer  
{  	
    cv::Mat buffer[SAVE_IMAGE_BUFFER_SIZE];
    long buffer_stamp[SAVE_IMAGE_BUFFER_SIZE];
    pthread_mutex_t lock; /* 互斥体lock 用于对缓冲区的互斥操作 */  
    int readpos, writepos; /* 读写指针*/  
    pthread_cond_t notempty; /* 缓冲区非空的条件变量 */  
    pthread_cond_t notfull; /* 缓冲区未满的条件变量 */  
};

class SaveDataProcess
{
public:
    SaveDataProcess();
    ~SaveDataProcess();

    int init_data();

    int init_save_dir();

    int set_save_dir(const std::string &image_path, const std::string &tof_path);

    std::string get_image_save_dir();

    int start();
    int stop();

    int video_start();
    int video_stop();

    int offline_start();
    int offline_stop();

    void put_image_data(cv::Mat &src_image, const long stamp);
    void put_tof_data(cv::Mat &depth_map, const long stamp);

    void get_image(cv::Mat &src_image);
    void get_image_yuv(cv::Mat &src_image);
    void get_tof_depth_map(cv::Mat &depth_map);

    void save_image(const cv::Mat &src_image, const long stamp);
    void save_image(const unsigned char *yuv_data, const long stamp);
    void save_depth_map(const cv::Mat &depth_map, const long stamp);
    void save_depth_map(const cv::Mat &depth_map, const std::string &save_path);
    void save_tof_z(const unsigned char* tof_data);

private:
    pthread_t image_pthread_id;
    pthread_t tof_pthread_id;
    pthread_t video_pthread_id;
    pthread_t offline_image_pthread_id;
    pthread_t offline_tof_pthread_id;
    std::string save_dir;
    unsigned long long int save_index;
    unsigned long long int tof_frame_number;
    unsigned long long int image_frame_number;
};

#endif // _SAVE_DATA_PROCESS_H_