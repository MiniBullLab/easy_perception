#ifndef _IMAGE_PROCESS_H_
#define _IMAGE_PROCESS_H_

#include <pthread.h>
#include <semaphore.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#define IMAGE_BUFFER_SIZE (40)

struct ImageBuffer  
{  	
	cv::Mat buffer[IMAGE_BUFFER_SIZE];
    pthread_mutex_t lock; /* 互斥体lock 用于对缓冲区的互斥操作 */  
    int readpos, writepos; /* 读写指针*/  
    pthread_cond_t notempty; /* 缓冲区非空的条件变量 */  
    pthread_cond_t notfull; /* 缓冲区未满的条件变量 */  
};

int put_image_buffer(cv::Mat &image_mat);

#endif // _IMAGE_PROCESS_H_