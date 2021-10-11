/*
  The following source code derives from Darknet
*/

#ifndef YOLOV3_H__
#define YOLOV3_H__

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <math.h>

#define YOLO_NET_IN_HEIGHT 416
#define YOLO_NET_IN_WIDTH 416
#define g_anchorCnt 3
#define g_classificationCnt 4
#define NMS_THRESHOLD 0.45f
#define CONF 0.24

const std::vector<float> g_anchors{8.95, 8.57, 12.43, 26.71, 19.71, 14.43,
                                 26.36, 58.52, 36.09, 25.55, 64.42, 42.90,
                                 96.44, 79.10, 158.37, 115.59, 218.65, 192.90};
// const std::vector<float> g_anchors{10,13, 16,30, 33,23,
//                                   30,61, 62,45, 59,119, 
//                                   116,90, 156,198, 373,326};


const int g_layer0_w = (YOLO_NET_IN_WIDTH / 32); //32)
const int g_layer0_h = (YOLO_NET_IN_HEIGHT / 32);   //32)

const int g_layer1_w = (YOLO_NET_IN_WIDTH / 16);     //16)
const int g_layer1_h = (YOLO_NET_IN_HEIGHT / 16);    //16)

const int g_layer2_w = (YOLO_NET_IN_WIDTH / 8);     //8)
const int g_layer2_h = (YOLO_NET_IN_HEIGHT / 8);    //8)

const int g_yolo_layer_channel = (g_anchorCnt *(g_classificationCnt+5));

std::vector<std::vector<float>> yolo_run(float* node0, float* node1, float* node2);

int entry_index(int loc, int anchorC, int w, int h, int lWidth, int lHeight);
float sigmoid(float p);
float overlap(float x1, float w1, float x2, float w2);
float cal_iou(std::vector<float> box, std::vector<float>truth);

//for amba version
void detect(std::vector<std::vector<float>> &boxes, const float* lOutput, \
			      int lHeight, int lWidth, int num, \
			      int sHeight, int sWidth);
std::vector<std::vector<float>> applyNMS(std::vector<std::vector<float>>& boxes,
							                           const float thres, bool sign_nms=true);

#endif  //YOLOV3_H__
