#ifndef _TOF_ACQUISITION_H_
#define _TOF_ACQUISITION_H_

#include <vector>
#include <pthread.h>

#include <glog/logging.h>
#include <glog/raw_logging.h>

//opencv
#include <opencv2/core.hpp>

class TOFAcquisition
{
public:
    struct Point
    {
        float x;
        float y;
        float z;
        int index;
    };
    typedef std::vector<Point> PointCloud;

    TOFAcquisition();
    ~TOFAcquisition();

    int open_tof();

    int start();
    int stop();

    void set_up();
    void set_sleep();

    void get_tof_data(PointCloud &point_cloud, cv::Mat &depth_map);

    int dump_ply(const char* save_path, const PointCloud &src_cloud);
    int dump_bin(const std::string &save_path, const PointCloud &src_cloud);
    int read_bin(const std::string &file_path, PointCloud &result_cloud);

private:
    pthread_t pthread_id;
};

#endif // _TOF_ACQUISITION_H_