#include "image_process.h"
#include <iostream>

struct ImageBuffer image_buffer;

int put_image_buffer(cv::Mat &image_mat)
{
	uint64_t start_time = 0;
	int rval = 0;
	if(image_mat.empty())
	{
		std::cout << "image empty!" << std::endl;
		return -1;
	}
	pthread_mutex_lock(&image_buffer.lock);  
    if ((image_buffer.writepos + 1) % IMAGE_BUFFER_SIZE == image_buffer.readpos)  
    {  
        pthread_cond_wait(&image_buffer.notfull, &image_buffer.lock);  
    }
	image_buffer.buffer[image_buffer.writepos] = image_mat.clone();
    image_buffer.writepos++;  
    if (image_buffer.writepos >= IMAGE_BUFFER_SIZE)  
        image_buffer.writepos = 0;  
    pthread_cond_signal(&image_buffer.notempty);  
    pthread_mutex_unlock(&image_buffer.lock);  
	std::cout << "put image" << std::endl;
	return rval;
}

// static void *save_video_pthread(void* save_data)
// {
// 	int rval = 0;
// 	bool first_save = true;
// 	unsigned long long int frame_number = 0;
// 	cv::Mat image_mat;
// 	cv::VideoWriter output_video;
// 	struct timeval tv;  
//     char time_str[64]; 
// 	while (run_flag > 0) 
// 	{
// 		while(run_lpr > 0)
// 		{
// 			pthread_mutex_lock(&image_buffer.lock);  
// 			if (image_buffer.writepos == image_buffer.readpos)  
// 			{  
// 				pthread_cond_wait(&image_buffer.notempty, &image_buffer.lock);  
// 			}
// 			image_mat = image_buffer.buffer[image_buffer.readpos];
// 			image_buffer.readpos++;  
// 			if (image_buffer.readpos >= IMAGE_BUFFER_SIZE)  
// 				image_buffer.readpos = 0; 
// 			pthread_cond_signal(&image_buffer.notfull);  
// 			pthread_mutex_unlock(&image_buffer.lock);
// 			if(has_lpr > 0 && first_save)
// 			{
// 				if (output_video.isOpened())
// 				{
// 					output_video.release();
// 				}
// 				std::stringstream filename;
// 				gettimeofday(&tv, NULL);  
// 				strftime(time_str, sizeof(time_str)-1, "%Y-%m-%d_%H:%M:%S", localtime(&tv.tv_sec)); 
// 				filename << "./result_video/" << time_str << ".avi";
// 				if(output_video.open(filename.str(), cv::VideoWriter::fourcc('X','V','I','D'), 25, \
// 					cv::Size(image_mat.cols, image_mat.rows)))
// 				{
// 					first_save = false;
// 				}
// 			}
// 			else if(has_lpr > 0)
// 			{
// 				if (output_video.isOpened())
// 				{
// 					output_video.write(image_mat);
// 				}
// 			}
// 			else
// 			{
// 				if (output_video.isOpened())
// 				{
// 					output_video.release();
// 					first_save = true;
// 				}
// 			}
// 			std::cout << "save image" << std::endl;
// 		}
// 		if (output_video.isOpened())
// 		{
// 			output_video.release();
// 			first_save = true;
// 		}
// 		usleep(20000);
// 		// std::cout << "save video runing" << std::endl;
// 	}
// 	if(output_video.isOpened())
//     {
//         output_video.release();
//     }
// 	std::cout << "stop save video pthread" << std::endl;
// 	return NULL;
// }