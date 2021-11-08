#ifndef _IMAGE_ACQUISITION_H_
#define _IMAGE_ACQUISITION_H_

#include <pthread.h>

#include <glog/logging.h>
#include <glog/raw_logging.h>

//opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#define IMAGE_BUFFER_SIZE (40)
#define IMAGE_WIDTH (1920)
#define IMAGE_HEIGHT (1080)

struct ImageBuffer  
{  	
	cv::Mat buffer[IMAGE_BUFFER_SIZE];
    pthread_mutex_t lock; /* 互斥体lock 用于对缓冲区的互斥操作 */  
    int readpos, writepos; /* 读写指针*/  
    pthread_cond_t notempty; /* 缓冲区非空的条件变量 */  
    pthread_cond_t notfull; /* 缓冲区未满的条件变量 */  
};

class ImageAcquisition
{
public:

    ImageAcquisition();
    ~ImageAcquisition();

    int init_camera();

    int start();
    int stop();

    void get_image(cv::Mat &src_image);

private:
    pthread_t pthread_id;
};

#endif // _IMAGE_ACQUISITION_H_