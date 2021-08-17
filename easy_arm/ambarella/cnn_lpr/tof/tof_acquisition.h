#ifndef _TOF_ACQUISITION_H_
#define _TOF_ACQUISITION_H_

#include <vector>
#include <pthread.h>

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

    void get_tof_data(PointCloud &point_cloud, cv::Mat &depth_map);

private:
    pthread_t pthread_id;
};

#endif // _TOF_ACQUISITION_H_