#ifndef _IMAGE_ACQUISITION_H_
#define _IMAGE_ACQUISITION_H_

#include <vector>
#include <pthread.h>

#include <glog/logging.h>
#include <glog/raw_logging.h>

//opencv
#include <opencv2/core.hpp>

class ImageAcquisition
{
public:

    ImageAcquisition();
    ~ImageAcquisition();

    int open_camera();

    int start();
    int stop();

private:
    pthread_t pthread_id;
};

#endif // _IMAGE_ACQUISITION_H_