#include "save_data_process.h"
#include "utility/utils.h"
#include <semaphore.h>
#include <iostream>
#include <fstream>
#include <algorithm> 

#define IS_USE_STAMP

static sem_t sem_put, sem_get;
static struct SaveImageBuffer image_buffer;
// static struct SaveImageBuffer video_buffer;
static struct SaveTofBuffer tof_buffer; 

static std::string save_image_dir = "./";
static std::string save_tof_dir = "./";

volatile int save_run = 0;
volatile int video_save_run = 0;

static std::vector<std::pair<long, std::string>> sort_path_list(const std::vector<std::string> &data_list, const std::string &post)
{
    std::pair<long, std::string> temp_data(0, "");
    std::vector<std::pair<long, std::string> > sort_list;
    int count = data_list.size();
    sort_list.clear();
    for (size_t index = 0; index < count; index++)
    {
        std::string::size_type iPos = data_list[index].find_last_of('/') + 1;
	    std::string filename = data_list[index].substr(iPos, data_list[index].length() - iPos);
        std::string name = filename.substr(0, filename.rfind("."));
        std::string suffix_str = filename.substr(filename.find_last_of('.') + 1);
        // std::cout << data_list[index] << " " << name << " " << suffix_str << std::endl;
        if(suffix_str == post)
        {
            long stamp = std::stol(name);
            temp_data.first = stamp;
            temp_data.second = data_list[index];
            sort_list.push_back(temp_data);
        }
    }

    std::sort(sort_list.begin(), sort_list.end(),[&](std::pair<long, std::string> a, std::pair<long, std::string> b){
        return a.first < b.first;
    });

    return sort_list;
}

// static bool saveMapBin(const std::string& filePath, const cv::Mat& map)
// {
// 	if (map.empty())
// 		return false;
 
// 	const char* filenamechar = filePath.c_str();
// 	FILE* fpw = fopen(filenamechar, "wb");//如果没有则创建，如果存在则从头开始写
// 	if (fpw == NULL)
// 	{
// 		//不可取fclose(fpw);
// 		return false;
// 	}
 
// 	int chan = map.channels();//用1个字节存,通道
// 	int type = map.type();//用2个字节存,类型,eg.CV_16SC2,CV_16UC1,CV_8UC1...
// 	int rows = map.rows;//用4个字节存,行数
// 	int cols = map.cols;//用4个字节存,列数
 
// 	fwrite(&chan, sizeof(char), 1, fpw);
// 	fwrite(&type, sizeof(char), 2, fpw);
// 	fwrite(&rows, sizeof(char), 4, fpw);
// 	fwrite(&cols, sizeof(char), 4, fpw);
 
// 	if (chan == 3)
// 	{
// 		if (type == CV_8UC3)//8U代表8位无符号整形,C3代表三通道
// 		{
// 			uchar* pData = (uchar*)map.data;
// 			for (int i = 0; i < rows * cols; i++)
// 			{
// 				fwrite(&pData[i * 3], sizeof(uchar), 1, fpw);
// 				fwrite(&pData[i * 3 + 1], sizeof(uchar), 1, fpw);
// 				fwrite(&pData[i * 3 + 2], sizeof(uchar), 1, fpw);
// 			}
// 		}
// 		else
// 		{
// 			fclose(fpw);
// 			return false;
// 		}
// 	}
// 	else if (chan == 2)
// 	{
// 		if (type == CV_8UC2)
// 		{
// 			uchar* pData = (uchar*)map.data;
// 			for (int i = 0; i < rows * cols; i++)
// 			{
// 				fwrite(&pData[i * 2], sizeof(uchar), 1, fpw);
// 				fwrite(&pData[i * 2 + 1], sizeof(uchar), 1, fpw);
// 			}
// 		}
// 		else if (type == CV_16SC2)//16S代表16位有符号整形,C2代表双通道
// 		{
// 			short* pData = (short*)map.data;
// 			for (int i = 0; i < rows * cols; i++)
// 			{
// 				fwrite(&pData[i * 2], sizeof(short), 1, fpw);
// 				fwrite(&pData[i * 2 + 1], sizeof(short), 1, fpw);
// 			}
// 		}
// 		else
// 		{
// 			fclose(fpw);
// 			return false;
// 		}
// 	}
// 	else if (chan == 1)
// 	{
// 		if (type == CV_8UC1)
// 		{
// 			uchar* pData = (uchar*)map.data;
// 			for (int i = 0; i < rows * cols; i++)
// 			{
// 				fwrite(&pData[i], sizeof(uchar), 1, fpw);
// 			}
// 		}
// 		else if (type == CV_16UC1)//16U代表16位无符号整形,C1代表单通道
// 		{
// 			ushort* pData = (ushort*)map.data;
// 			for (int i = 0; i < rows * cols; i++)
// 			{
// 				fwrite(&pData[i], sizeof(ushort), 1, fpw);
// 			}
// 		}
// 		else
// 		{
// 			fclose(fpw);
// 			return false;
// 		}
// 	}
// 	else
// 	{
// 		fclose(fpw);
// 		return false;
// 	}
 
// 	fclose(fpw);
// 	return true;
// }
 
// static bool loadMapBin(const std::string& filePath, cv::Mat& map)
// {
// 	const char* filenamechar = filePath.c_str();
// 	FILE* fpr = fopen(filenamechar, "rb");
// 	if (fpr == NULL)
// 	{
// 		//不可取fclose(fpr);
// 		return false;
// 	}
 
// 	int chan = 0;
// 	int type = 0;
// 	int rows = 0;
// 	int cols = 0;
 
// 	fread(&chan, sizeof(char), 1, fpr);//1个字节存,通道
// 	fread(&type, sizeof(char), 2, fpr);//2个字节存,类型,eg.CV_16SC2,CV_16UC1,CV_8UC1...
// 	fread(&rows, sizeof(char), 4, fpr);//4个字节存,行数
// 	fread(&cols, sizeof(char), 4, fpr);//4个字节存,列数
 
// 	map = cv::Mat::zeros(rows, cols, type);
 
// 	if (chan == 3)
// 	{
// 		if (type == CV_8UC3)//8U代表8位无符号整形,C3代表三通道
// 		{
// 			uchar* pData = (uchar*)map.data;
// 			for (int i = 0; i < rows * cols; i++)
// 			{
// 				fread(&pData[i * 3], sizeof(uchar), 1, fpr);
// 				fread(&pData[i * 3 + 1], sizeof(uchar), 1, fpr);
// 				fread(&pData[i * 3 + 2], sizeof(uchar), 1, fpr);
// 			}
// 		}
// 		else
// 		{
// 			fclose(fpr);
// 			return false;
// 		}
// 	}
// 	else if (chan == 2)
// 	{
// 		if (type == CV_8UC2)
// 		{
// 			uchar* pData = (uchar*)map.data;
// 			for (int i = 0; i < rows * cols; i++)
// 			{
// 				fread(&pData[i * 2], sizeof(uchar), 1, fpr);
// 				fread(&pData[i * 2 + 1], sizeof(uchar), 1, fpr);
// 			}
// 		}
// 		else if (type == CV_16SC2)//16S代表16位有符号整形,C2代表双通道
// 		{
// 			short* pData = (short*)map.data;
// 			for (int i = 0; i < rows * cols; i++)
// 			{
// 				fread(&pData[i * 2], sizeof(short), 1, fpr);
// 				fread(&pData[i * 2 + 1], sizeof(short), 1, fpr);
// 			}
// 		}
// 		else
// 		{
// 			fclose(fpr);
// 			return false;
// 		}
// 	}
// 	else if (chan == 1)
// 	{
// 		if (type == CV_8UC1)
// 		{
// 			uchar* pData = (uchar*)map.data;
// 			for (int i = 0; i < rows * cols; i++)
// 			{
// 				fread(&pData[i], sizeof(uchar), 1, fpr);
// 			}
// 		}
// 		else if (type == CV_16UC1)//16U代表16位无符号整形,C1代表单通道
// 		{
// 			ushort* pData = (ushort*)map.data;
// 			for (int i = 0; i < rows * cols; i++)
// 			{
// 				fread(&pData[i], sizeof(ushort), 1, fpr);
// 			}
// 		}
// 		else
// 		{
// 			fclose(fpr);
// 			return false;
// 		}
// 	}
// 	else
// 	{
// 		fclose(fpr);
// 		return false;
// 	}
 
// 	fclose(fpr);
// 	return true;
// }

// static void *save_video_pthread(void* save_data)
// {
// 	int rval = 0;
// 	bool first_save = true;
// 	unsigned long long int frame_number = 0;
// 	cv::Mat image_mat;
// 	cv::VideoWriter output_video;
// 	struct timeval tv;  
//     char time_str[64]; 
// 	while (video_save_run > 0) 
// 	{
//         pthread_mutex_lock(&video_buffer.lock);  
//         if (video_buffer.writepos == video_buffer.readpos)  
//         {  
//             pthread_cond_wait(&video_buffer.notempty, &video_buffer.lock);  
//         }
//         image_mat = video_buffer.buffer[video_buffer.readpos];
//         video_buffer.readpos++;  
//         if (video_buffer.readpos >= SAVE_IMAGE_BUFFER_SIZE)  
//             video_buffer.readpos = 0; 
//         pthread_cond_signal(&video_buffer.notfull);  
//         pthread_mutex_unlock(&video_buffer.lock);
//         if(first_save)
//         {
//             if (output_video.isOpened())
//             {
//                 output_video.release();
//             }
//             std::stringstream filename;
//             gettimeofday(&tv, NULL);  
//             strftime(time_str, sizeof(time_str)-1, "%Y-%m-%d_%H:%M:%S", localtime(&tv.tv_sec)); 
//             filename << "./result_video/" << time_str << ".avi";
//             if(output_video.open(filename.str(), cv::VideoWriter::fourcc('X','V','I','D'), 25, \
//                 cv::Size(image_mat.cols, image_mat.rows)))
//             {
//                 output_video.write(image_mat);
//                 first_save = false;
//             }
//         }
//         else
//         {
//             if (output_video.isOpened())
//             {
//                 output_video.write(image_mat);
//             }
//         }
// 		// std::cout << "save video runing" << std::endl;
// 	}
// 	if(output_video.isOpened())
//     {
//         output_video.release();
//     }
//     LOG(WARNING) << "stop save video pthread.";
// 	return NULL;
// }

static void *save_image_pthread(void* save_data)
{
    unsigned long long int frame_number = 0;
    long stamp = 0;
    prctl(PR_SET_NAME, "save_image_pthread");
    while(save_run) {
        pthread_mutex_lock(&image_buffer.lock);  
        if (image_buffer.writepos == image_buffer.readpos)  
        {  
            pthread_cond_wait(&image_buffer.notempty, &image_buffer.lock);  
        }
        cv::Mat src_image = image_buffer.buffer[image_buffer.readpos];
        stamp = image_buffer.buffer_stamp[image_buffer.readpos];
        std::stringstream filename_image;
        filename_image << save_image_dir << stamp << ".jpg";
        if(!src_image.empty())
        {
            cv::imwrite(filename_image.str(), src_image);
            //frame_number++;
        }
        else
        {
            LOG(ERROR) << "save src image empty: " << filename_image.str();
        }
        image_buffer.readpos++;
        if (image_buffer.readpos >= SAVE_IMAGE_BUFFER_SIZE)  
            image_buffer.readpos = 0; 
        pthread_cond_signal(&image_buffer.notfull);  
        pthread_mutex_unlock(&image_buffer.lock);
	}
	save_run = 0;
	LOG(WARNING) << "save image thread quit.";
    return NULL;
}

static void *save_tof_pthread(void* save_data)
{
    unsigned long long int frame_number = 0;
    long stamp = 0;
    prctl(PR_SET_NAME, "save_tof_pthread");
    while(save_run) {
		pthread_mutex_lock(&tof_buffer.lock);  
        if (tof_buffer.writepos == tof_buffer.readpos)  
        {  
            pthread_cond_wait(&tof_buffer.notempty, &tof_buffer.lock);  
        }
        cv::Mat depth_map = tof_buffer.buffer[tof_buffer.readpos];
        stamp = tof_buffer.buffer_stamp[image_buffer.readpos];
        std::stringstream filename_tof;
        filename_tof << save_tof_dir << stamp << ".bin";
        if(!depth_map.empty())
        {
            std::ofstream outF(filename_tof.str(), std::ios::binary);
            outF.write(reinterpret_cast<char*>(depth_map.data), DEPTH_WIDTH * DEPTH_HEIGTH * sizeof(uchar));
            outF.close();
            // cv::imwrite(filename_tof.str(), depth_map);
            // frame_number++;
        }
        else
        {
            LOG(ERROR) << "save depth map empty: " << filename_tof.str();
        }
        tof_buffer.readpos++;  
        if (tof_buffer.readpos >= SAVE_TOF_BUFFER_SIZE)  
            tof_buffer.readpos = 0; 
        pthread_cond_signal(&tof_buffer.notfull);  
        pthread_mutex_unlock(&tof_buffer.lock);
	}
	save_run = 0;
	LOG(WARNING) << "save tof thread quit.";
    return NULL;
}

static long tof_stamp = 0;
static int offline_image_count = 0;

static void *offline_image_pthread(void* save_data)
{
    unsigned long time_start, time_end;
    std::vector<std::string> data_list;
    std::vector<std::pair<long, std::string>> sort_list;
    size_t image_count = 0;
    long pre_stamp = 0;
    int sleep_time = 0;
    int base_time = 0;
    ListImages(save_image_dir, data_list);
    image_count = data_list.size();
    offline_image_count = image_count;
    if(image_count == 0)
    {
        LOG(ERROR) << "offline image data not exist:" << save_image_dir;
        return NULL;
    }

#if defined(IS_USE_STAMP)
    sort_list = sort_path_list(data_list, "jpg");
    image_count = sort_list.size();
#endif

    prctl(PR_SET_NAME, "offline_image_pthread");
    for (size_t index = 0; index < image_count && save_run > 0; index++) 
    {
        time_start = gettimeus();
        cv::Mat src_img;
		std::stringstream temp_str;
#if defined(IS_USE_STAMP)
        if(pre_stamp == 0)
        {
            pre_stamp = sort_list[index].first;
        }
        else
        {
            base_time = sort_list[index].first - pre_stamp;
            LOG(WARNING) << "image time diff:" << base_time;
            pre_stamp = sort_list[index].first;
        }
        // if(sem_wait(&sem_get) == 0)
        // {
        //     std::cout << "dgfdhgfh:" << tof_stamp << " "  << std::abs(sort_list[index].first - tof_stamp) << std::endl;
        //     if(std::abs(sort_list[index].first - tof_stamp) > 400)
        //     {
        //         sem_post(&sem_put);
        //         continue;
        //     }
        //     sem_post(&sem_put);
        // }
        temp_str << save_image_dir << sort_list[index].second;
#else
        temp_str << save_image_dir << "image_" << index << ".jpg";
#endif
        LOG(WARNING) << temp_str.str();
        pthread_mutex_lock(&image_buffer.lock);  
		if ((image_buffer.writepos + 1) % SAVE_IMAGE_BUFFER_SIZE == image_buffer.readpos)  
		{  
			pthread_cond_wait(&image_buffer.notfull, &image_buffer.lock);  
		}
		image_buffer.buffer[image_buffer.writepos] = cv::imread(temp_str.str());
        if(!image_buffer.buffer[image_buffer.writepos].empty())
        {
            image_buffer.writepos++;
        }
        else
        {
            LOG(ERROR) << "read image fail: " << temp_str.str();
        }
        if (image_buffer.writepos >= SAVE_IMAGE_BUFFER_SIZE)  
			image_buffer.writepos = 0;  
		pthread_cond_signal(&image_buffer.notempty);  
		pthread_mutex_unlock(&image_buffer.lock); 
        time_end = gettimeus();
        sleep_time = (base_time * 1000) - (time_end - time_start);
        LOG(INFO) << "offline get image cost time: " <<  (time_end - time_start)/1000.0  << "ms  sleep:" << sleep_time;
        if(sleep_time <= 0)
            sleep_time = 0;
        usleep(sleep_time);
        LOG(WARNING) << "offline image all cost time: " <<  (gettimeus() - time_start)/1000.0  << "ms";
    }
	LOG(WARNING) << "offline image thread quit.";
    return NULL;
}

static void *offline_tof_pthread(void* save_data)
{
    long pre_stamp = 0;
    unsigned long time_start, time_end;
    std::vector<std::string> data_list;
    std::vector<std::pair<long, std::string>> sort_list;
    ListImages(save_tof_dir, data_list);
    int sleep_time = 0;
    size_t tof_count = data_list.size() / 2 + 1;
    int base_time = 0;
    float count_ratio = 0;
    if(tof_count == 0)
    {
        LOG(ERROR) << "offline tof data not exist:" << save_tof_dir;
        return NULL;
    }
#if defined(IS_USE_STAMP)
    sort_list = sort_path_list(data_list, "bin");
    tof_count = sort_list.size();
#else
    count_ratio = (float)offline_image_count / tof_count;
    if(count_ratio < 1.1)
    {
        base_time = 110000;
    }
    else
    {
        base_time = 140000;
    }
    // base_time = (int)(count_ratio * 105000);
#endif
    
    prctl(PR_SET_NAME, "offline_tof_pthread");
    for (size_t index = 0; index < tof_count && save_run > 0; index++) 
    {
        time_start = gettimeus();
		std::stringstream temp_str;
#if defined(IS_USE_STAMP)
        if(pre_stamp == 0)
        {
            pre_stamp = sort_list[index].first;
        }
        else
        {
            base_time = sort_list[index].first - pre_stamp;
            LOG(WARNING) << "tof time diff:" << base_time;
            pre_stamp = sort_list[index].first;
        }
        // struct timespec ts;
        // clock_gettime(CLOCK_REALTIME, &ts);
        // ts.tv_sec += 1;
        // ts.tv_nsec = 0;
        // //sem_timedwait(&sem_put, &ts);
        // if(sem_wait(&sem_put) == 0)
        // {
        //     tof_stamp = sort_list[index].first;
        //     sem_post(&sem_get);
        // }
        temp_str << save_tof_dir << sort_list[index].second;
#else
        temp_str << save_tof_dir << "tof_" << index << ".bin";
#endif
        LOG(WARNING) << temp_str.str();
        pthread_mutex_lock(&tof_buffer.lock);  
		if ((tof_buffer.writepos + 1) % SAVE_TOF_BUFFER_SIZE == tof_buffer.readpos)  
		{  
			pthread_cond_wait(&tof_buffer.notfull, &tof_buffer.lock);  
		}
        std::ifstream in_file(temp_str.str(), std::ios::in|std::ios::binary);
        if (in_file.is_open())
        {
           cv::Mat mat(DEPTH_HEIGTH, DEPTH_WIDTH, CV_8UC1);
        //    uchar* pData = (uchar*)mat.data;
		//    for (int i = 0; i < DEPTH_HEIGTH * DEPTH_WIDTH; i++)
		//    {
        //        in_file.read(reinterpret_cast<char*>(&pData[i]), sizeof(uchar));
		//    }
           in_file.read(reinterpret_cast<char*>(mat.data), CV_ELEM_SIZE(CV_8UC1) * DEPTH_HEIGTH * DEPTH_WIDTH);
        //    std::stringstream filename_tof;
        //    filename_tof << "/data/save_data/" << "tof_" << index << ".jpg";
        //    cv::imwrite(filename_tof.str(), mat);
           tof_buffer.buffer[tof_buffer.writepos] = mat;
           tof_buffer.writepos++;
        }
        else
        {
            LOG(ERROR) << "read tof fail: " << temp_str.str();
        }
        in_file.close();
        if (tof_buffer.writepos >= SAVE_TOF_BUFFER_SIZE)  
			tof_buffer.writepos = 0;  
		pthread_cond_signal(&tof_buffer.notempty);  
		pthread_mutex_unlock(&tof_buffer.lock); 
        time_end = gettimeus();
        if(base_time > 0)
            sleep_time = (base_time * 1000) - (time_end - time_start);
        LOG(INFO) << "offline get tof cost time: " <<  (time_end - time_start)/1000.0  << "ms sleep:" << sleep_time;
        if(sleep_time <= 0)
            sleep_time = 160;
        usleep(sleep_time);
        LOG(WARNING) << "offline tof all cost time: " <<  (gettimeus() - time_start)/1000.0  << "ms";
    }
	LOG(WARNING) << "offline tof thread quit.";
    return NULL;
}

SaveDataProcess::SaveDataProcess()
{
    image_pthread_id = 0;
    tof_pthread_id = 0;
    video_pthread_id = 0;
    offline_image_pthread_id = 0;
    offline_tof_pthread_id = 0;

    save_run = 0;
    video_save_run = 0;

    pthread_mutex_init(&image_buffer.lock, NULL);  
    pthread_cond_init(&image_buffer.notempty, NULL);  
    pthread_cond_init(&image_buffer.notfull, NULL);  
    image_buffer.readpos = 0;  
    image_buffer.writepos = 0;

    pthread_mutex_init(&tof_buffer.lock, NULL);  
    pthread_cond_init(&tof_buffer.notempty, NULL);  
    pthread_cond_init(&tof_buffer.notfull, NULL);  
    tof_buffer.readpos = 0;  
    tof_buffer.writepos = 0;

    save_dir = "/data/save_data/";
    save_index = 0;

    tof_frame_number = 0;
    image_frame_number = 0;

    // sem_init(&sem_put, 0, 1);
	// sem_init(&sem_get, 0, 0);

    LOG(WARNING) << "tof:" << SAVE_TOF_BUFFER_SIZE << " image:" << SAVE_IMAGE_BUFFER_SIZE;
}

SaveDataProcess::~SaveDataProcess()
{
#ifndef ONLY_SAVE_DATA
    if(save_run > 0)
	{
		stop();
	}
#endif
    pthread_mutex_destroy(&image_buffer.lock);
    pthread_cond_destroy(&image_buffer.notempty);
    pthread_cond_destroy(&image_buffer.notfull);

    pthread_mutex_destroy(&tof_buffer.lock);
    pthread_cond_destroy(&tof_buffer.notempty);
    pthread_cond_destroy(&tof_buffer.notfull);

    // sem_destroy(&sem_put);
	// sem_destroy(&sem_get);

    LOG(WARNING) << "~SaveDataProcess()";
}

int SaveDataProcess::init_data()
{
    std::vector<std::string> dir_list;
    ListPath(save_dir, dir_list);
    for (size_t index = 0; index < dir_list.size(); index++) 
    {
        size_t pos = dir_list[index].find_last_of('_');
        LOG(WARNING) << dir_list[index] << " " << pos;
        if(pos >= 0)
        {
            std::string temp = dir_list[index].substr(pos + 1);
            int temp_index = atoi(temp.c_str());
            if(temp_index > save_index)
            {
                save_index = temp_index;
            }
        }
    }
    save_index++;
    tof_frame_number = 0;
    image_frame_number = 0;
    LOG(WARNING) << "index:" <<save_index;
    return 0;
}

int SaveDataProcess::init_save_dir()
{
    struct timeval tv;  
    char time_str[64];
	std::stringstream image_save_path;
    std::stringstream tof_save_path;
    std::string command;
    gettimeofday(&tv, NULL); 
	strftime(time_str, sizeof(time_str)-1, "%Y_%m_%d_%H_%M_%S", localtime(&tv.tv_sec)); 
	image_save_path << save_dir << time_str << "_" << save_index << "/image/";
    tof_save_path << save_dir << time_str << "_" << save_index << "/tof/";
    save_image_dir = image_save_path.str();
    save_tof_dir = tof_save_path.str();
    command = "mkdir -p " + save_image_dir;
    system(command.c_str());
    command = "mkdir -p " + save_tof_dir;
    system(command.c_str());
    // if(!system(command.c_str()))
    // {
    //     save_run = 0;
    //     LOG(ERROR) << "mkdir dir fail:" << command;
    //     return -1;
    // }
    // if (0 != access(save_path.str().c_str(), 0))
    // {
    //     // if this folder not exist, create a new one.
    //     mkdir(save_path.str().c_str());   // 返回 0 表示创建成功，-1 表示失败
    // }
    // else
    // {
    //     save_run = 0;
    //     LOG(ERROR) << "mkdir dir fail:" << command;
    //     return -1;
    // }
    return 0;
}

int SaveDataProcess::set_save_dir(const std::string &image_path, const std::string &tof_path)
{
    save_image_dir = image_path;
    save_tof_dir = tof_path;
    return 0;
}

std::string SaveDataProcess::get_image_save_dir()
{
    return save_image_dir;
}

int SaveDataProcess::start()
{
    int ret = 0;
    ret = init_save_dir();

    save_run = 1;

    image_buffer.readpos = 0;  
    image_buffer.writepos = 0;

    tof_buffer.readpos = 0;  
    tof_buffer.writepos = 0;

    ret = pthread_create(&image_pthread_id, NULL, save_image_pthread, NULL);
    if(ret < 0)
    {
        save_run = 0;
        LOG(ERROR) << "save image pthread fail!";
    }
    else
    {
        LOG(WARNING) << "start image pthread:" << image_pthread_id;
        ret = pthread_create(&tof_pthread_id, NULL, save_tof_pthread, NULL);
        if(ret < 0)
        {
            save_run = 0;
            LOG(ERROR) << "save tof pthread fail!";
        }
        LOG(WARNING) << "start save tof pthread:" << tof_pthread_id;
    }
    LOG(INFO) << "save pthread start success!";
	return ret;
}

int SaveDataProcess::stop()
{
    int ret = 0;
	save_run = 0;

    // LOG(WARNING) << "stop save data";

	if (image_pthread_id > 0) {
        pthread_cond_signal(&image_buffer.notempty);  
        pthread_mutex_unlock(&image_buffer.lock);
		pthread_join(image_pthread_id, NULL);
        image_pthread_id = 0;
	}
    if (tof_pthread_id > 0) {
        pthread_cond_signal(&tof_buffer.notempty);  
        pthread_mutex_unlock(&tof_buffer.lock);
		pthread_join(tof_pthread_id, NULL);
        tof_pthread_id = 0;
	}
	LOG(WARNING) << "stop save data success";
	return ret;
}

int SaveDataProcess::video_start()
{
    return 0;
}

int SaveDataProcess::video_stop()
{
    return 0;
}

int SaveDataProcess::offline_start()
{
    int ret = 0;
    
    save_run = 1;

    image_buffer.readpos = 0;  
    image_buffer.writepos = 0;

    tof_buffer.readpos = 0;  
    tof_buffer.writepos = 0;

    ret = pthread_create(&offline_tof_pthread_id, NULL, offline_tof_pthread, NULL);
    if(ret < 0)
    {
        save_run = 0;
        LOG(ERROR) << "offline tof pthread fail!";
    }
    else
    {
        LOG(WARNING) << "start offline tof pthread:" << offline_tof_pthread_id;
        ret = pthread_create(&offline_image_pthread_id, NULL, offline_image_pthread, NULL);
        if(ret < 0)
        {
            save_run = 0;
            LOG(ERROR) << "offline image pthread fail!";
        }
        LOG(WARNING) << "start offline image pthread:" << offline_image_pthread_id;
    }
    LOG(INFO) << "offline pthread start success!";
	return ret;
}

int SaveDataProcess::offline_stop()
{
    int ret = 0;
	save_run = 0;

    // LOG(WARNING) << "stop offline data";

    // sem_post(&sem_put);
    // sem_post(&sem_get);

	if (offline_image_pthread_id > 0) {
        pthread_cond_signal(&image_buffer.notfull);
        pthread_cond_signal(&image_buffer.notempty);  
        pthread_mutex_unlock(&image_buffer.lock);
		pthread_join(offline_image_pthread_id, NULL);
        offline_image_pthread_id = 0;
	}
    if (offline_tof_pthread_id > 0) {
        pthread_cond_signal(&tof_buffer.notfull);
        pthread_cond_signal(&tof_buffer.notempty);  
        pthread_mutex_unlock(&tof_buffer.lock);
		pthread_join(offline_tof_pthread_id, NULL);
        offline_tof_pthread_id = 0;
	}
	LOG(WARNING) << "stop offline data success";
	return ret;
}

void SaveDataProcess::put_image_data(cv::Mat &src_image, const long stamp)
{
    if (image_pthread_id > 0 &&  save_run > 0) 
    {
        if(!src_image.empty())
        {
            pthread_mutex_lock(&image_buffer.lock);  
            if ((image_buffer.writepos + 1) % SAVE_IMAGE_BUFFER_SIZE == image_buffer.readpos)  
            {  
                pthread_cond_wait(&image_buffer.notfull, &image_buffer.lock);  
            }
            image_buffer.buffer[image_buffer.writepos] = src_image.clone();
            image_buffer.buffer_stamp[tof_buffer.writepos] = stamp;
            image_buffer.writepos++;  
            if (image_buffer.writepos >= SAVE_IMAGE_BUFFER_SIZE)  
                image_buffer.writepos = 0;  
            pthread_cond_signal(&image_buffer.notempty);  
            pthread_mutex_unlock(&image_buffer.lock);
        }
        else
        {
            LOG(ERROR) << "put src image is empty!";
        }
    } 
}

void SaveDataProcess::put_tof_data(cv::Mat &depth_map, const long stamp)
{
    if(tof_pthread_id > 0 &&  save_run > 0)
    {
        if(!depth_map.empty())
        {
            pthread_mutex_lock(&tof_buffer.lock);  
            if ((tof_buffer.writepos + 1) % SAVE_TOF_BUFFER_SIZE == tof_buffer.readpos)  
            {  
                pthread_cond_wait(&tof_buffer.notfull, &tof_buffer.lock);  
            }
            tof_buffer.buffer[tof_buffer.writepos] = depth_map.clone();
            tof_buffer.buffer_stamp[tof_buffer.writepos] = stamp;
            tof_buffer.writepos++;  
            if (tof_buffer.writepos >= SAVE_TOF_BUFFER_SIZE)  
                tof_buffer.writepos = 0;  
            pthread_cond_signal(&tof_buffer.notempty);  
            pthread_mutex_unlock(&tof_buffer.lock);
        }
        else
        {
            LOG(ERROR) << "put depth map is empty!";
        }
    }  
}

void SaveDataProcess::get_image(cv::Mat &src_image)
{
    if(offline_image_pthread_id > 0)
	{
		pthread_mutex_lock(&image_buffer.lock);  
		if (image_buffer.writepos == image_buffer.readpos)  
		{  
			pthread_cond_wait(&image_buffer.notempty, &image_buffer.lock);  
		}
        src_image = image_buffer.buffer[image_buffer.readpos].clone();
		image_buffer.readpos++;  
		if (image_buffer.readpos >= SAVE_IMAGE_BUFFER_SIZE)  
			image_buffer.readpos = 0; 
		pthread_cond_signal(&image_buffer.notfull);  
		pthread_mutex_unlock(&image_buffer.lock);
	}
}

void SaveDataProcess::get_image_yuv(cv::Mat &src_image)
{
    if(offline_image_pthread_id > 0)
	{
		pthread_mutex_lock(&image_buffer.lock);  
		if (image_buffer.writepos == image_buffer.readpos)  
		{  
			pthread_cond_wait(&image_buffer.notempty, &image_buffer.lock);  
		}
    #if CV_VERSION_MAJOR < 4
		cv::cvtColor(image_buffer.buffer[image_buffer.readpos], src_image, CV_BGR2YUV_IYUV);
	#else
		cv::cvtColor(image_buffer.buffer[image_buffer.readpos], src_image, cv::COLOR_BGR2YUV_IYUV);
	#endif
		image_buffer.readpos++;  
		if (image_buffer.readpos >= SAVE_IMAGE_BUFFER_SIZE)  
			image_buffer.readpos = 0; 
		pthread_cond_signal(&image_buffer.notfull);  
		pthread_mutex_unlock(&image_buffer.lock);
	}
}

void SaveDataProcess::get_tof_depth_map(cv::Mat &depth_map)
{
    if(offline_tof_pthread_id > 0) 
	{
		pthread_mutex_lock(&tof_buffer.lock);  
		if (tof_buffer.writepos == tof_buffer.readpos)  
		{  
			pthread_cond_wait(&tof_buffer.notempty, &tof_buffer.lock);  
		}
        depth_map = tof_buffer.buffer[tof_buffer.readpos].clone();
		tof_buffer.readpos++;  
		if (tof_buffer.readpos >= SAVE_TOF_BUFFER_SIZE)  
			tof_buffer.readpos = 0; 
		pthread_cond_signal(&tof_buffer.notfull);  
		pthread_mutex_unlock(&tof_buffer.lock);
	}
}

void SaveDataProcess::save_image(const cv::Mat &src_image, const long stamp)
{
    unsigned long time_start;
    std::stringstream filename_image;
    time_start = gettimeus();
    filename_image << save_image_dir << stamp << ".jpg";
    if(!src_image.empty())
    {
        std::vector<int> compression_params;
	    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
	    compression_params.push_back(90);
        cv::imwrite(filename_image.str(), src_image, compression_params);
        // image_frame_number++;
    }
    else
    {
        LOG(ERROR) << "save src image empty: " << filename_image.str();
    }
    LOG(WARNING) << "save image cost time: " <<  (gettimeus() - time_start)/1000.0  << "ms";
}

void SaveDataProcess::save_image(const unsigned char *yuv_data, const long stamp)
{
    unsigned long time_start;
    std::stringstream filename_image;
    time_start = gettimeus();
    filename_image << save_image_dir << stamp << ".yuv";
    if(yuv_data != NULL)
    {
        std::ofstream outF(filename_image.str(), std::ios::binary);
        outF.write(reinterpret_cast<const char*>(yuv_data),  IMAGE_YUV_SIZE * sizeof(uchar));
        outF.close();
        // image_frame_number++;
    }
    else
    {
        LOG(ERROR) << "save src image empty: " << filename_image.str();
    }
    LOG(WARNING) << "save image cost time: " <<  (gettimeus() - time_start)/1000.0  << "ms";
}

void SaveDataProcess::save_depth_map(const cv::Mat &depth_map, const long stamp)
{
    std::stringstream filename_tof;
    std::stringstream temp_tof;
    filename_tof << save_tof_dir << stamp << ".bin";
    temp_tof << save_tof_dir << stamp << ".jpg";
    if(!depth_map.empty())
    {
        std::vector<int> compression_params;
	    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
	    compression_params.push_back(90);
        std::ofstream outF(filename_tof.str(), std::ios::binary);
        outF.write(reinterpret_cast<char*>(depth_map.data), DEPTH_WIDTH * DEPTH_HEIGTH * sizeof(uchar));
        outF.close();
        cv::imwrite(temp_tof.str(), depth_map, compression_params);
        // tof_frame_number++;
    }
    else
    {
        LOG(ERROR) << "save depth map empty: " << filename_tof.str();
    }
}

void SaveDataProcess::save_depth_map(const cv::Mat &depth_map, const std::string &save_path)
{
    if(!depth_map.empty())
    {
        std::vector<int> compression_params;
	    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
	    compression_params.push_back(90);
        // std::ofstream outF(save_path, std::ios::binary);
        // outF.write(reinterpret_cast<char*>(depth_map.data), DEPTH_WIDTH * DEPTH_HEIGTH * sizeof(uchar));
        // outF.close();
        cv::imwrite(save_path, depth_map, compression_params);
        // tof_frame_number++;
    }
    else
    {
        LOG(ERROR) << "save depth map empty: " << save_path;
    }
}

void SaveDataProcess::save_tof_z(const unsigned char* tof_data)
{
    std::stringstream filename_tof;
    filename_tof << save_tof_dir << get_time_stamp() << ".bin";
    std::ofstream outF(filename_tof.str(), std::ios::binary);
    outF.write(reinterpret_cast<const char*>(tof_data), TOF_SIZE * sizeof(unsigned char));
    outF.close();
    // tof_frame_number++;
}