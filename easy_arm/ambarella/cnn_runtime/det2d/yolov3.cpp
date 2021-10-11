/*
  The following source code derives from Darknet
*/

#include "cnn_runtime/det2d/yolov3.h"
#include "cnn_runtime/cnn_common/cnn_function.h"
#include "cnn_runtime/cnn_common/blob_define.h"

#define IDX(o) (entry_index(o,c,w,h,lWidth,lHeight))


int entry_index(int loc, int anchorC, int w, int h, int lWidth, int lHeight)
{
    return ((anchorC *(g_classificationCnt+5) + loc) * lHeight * LAYER_P(lWidth) + h * LAYER_P(lWidth) + w);
}

float overlap(float x1, float w1, float x2, float w2)
{
    float left = std::max(x1 - w1 / 2.0f, x2 - w2 / 2.0f);
    float right = std::min(x1 + w1 / 2.0f, x2 + w2 / 2.0f);
    return right - left;
}

float cal_iou(std::vector<float> box, std::vector<float>truth)
{
    float w = overlap(box[0], box[2], truth[0], truth[2]);
    float h = overlap(box[1], box[3], truth[1], truth[3]);
    if(w < 0 || h < 0) return 0;

    float inter_area = w * h;
    float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
    return inter_area * 1.0f / union_area;
}

/**************************************
Function: detect (for AMBA version)
purpose: yolo layer for AMBA version

Parammeter:  
            1.boxes: detection objects to return
            2.lOutput: AMBA output
            3.lHeight,lWidth: feature map's height,width 
            4.num: layer num
            5.sHeight,sWidth: yolo net input height, width
Return: 
**************************************/
void detect(std::vector<std::vector<float>> &boxes, const float* lOutput,
            int lHeight, int lWidth, int num, int sHeight, int sWidth) 
{
    std::vector<float> box;
	float max;
    int cls = 0;
    for (int h = 0; h < lHeight; ++h) {
        for (int w = 0; w < lWidth; ++w) {
            for (int c = 0; c < g_anchorCnt; ++c) {
                float obj_score = sigmoid(lOutput[IDX(4)]);
                if (obj_score < CONF)
                    continue;

                max = -1e10;
                cls = 0;
				for (int p = 0; p < g_classificationCnt; p++) {
					if (lOutput[IDX(5 + p)] >= max) {
						max = lOutput[IDX(5 + p)];
						cls = p;
					}
				}
                
                max = sigmoid(max);
				if (max <= CONF) {
					continue;
				}

				obj_score *= max;
				if (obj_score <= CONF) {
					continue;
				}

                box.clear();

                box.push_back((w + sigmoid(lOutput[IDX(0)])) / lWidth);
                box.push_back((h + sigmoid(lOutput[IDX(1)])) / lHeight);
                box.push_back(exp(lOutput[IDX(2)]) * g_anchors[2 * (c + g_anchorCnt * num)] / float(sWidth));
                box.push_back(exp(lOutput[IDX(3)]) * g_anchors[2 * (c + g_anchorCnt * num) + 1] / float(sHeight));
                box.push_back((float) cls);
                box.push_back(obj_score);
                //for (int p = 0; p < g_classificationCnt; p++) {
                //    box.push_back(obj_score * sigmoid(lOutput[IDX(5 + p)]));
                //}
                boxes.push_back(box);
            }
        }
    }
    // std::cout << "yolo node boxes size is:  " << boxes.size() << std::endl;
}

/**************************************
Parammeter: 
            1.boxes: all detection objects
            2.classes: classes number(unuse in this version)
            3.thres: iou threshold
            4.sign_nms: do road sign nms or not, default is true

Return: detection objects after nms
**************************************/
std::vector<std::vector<float>> applyNMS(std::vector<std::vector<float>>& boxes,
	                                    const float thres, bool sign_nms) 
{    
    std::vector<std::vector<float>> result;
    std::vector<bool> exist_box(boxes.size(), true);

    int n = 0;
    for (size_t _i = 0; _i < boxes.size(); ++_i) 
	{
        if (!exist_box[_i]) 
			continue;
        n = 0;
        for (size_t _j = _i + 1; _j < boxes.size(); ++_j)
		{
            // different class name
            if (!exist_box[_j] || boxes[_i][4] != boxes[_j][4]) 
				continue;
            float ovr = cal_iou(boxes[_j], boxes[_i]);
            if (ovr >= thres) 
            {
                if (boxes[_j][5] <= boxes[_i][5])
                {
                    exist_box[_j] = false;
                }
                else
                {
                    n++;   // have object_score bigger than boxes[_i]
                    exist_box[_i] = false;
                    break;
                }
            }
        }
        //if (n) exist_box[_i] = false;
		if (n == 0) 
		{
			result.push_back(boxes[_i]);
		}			
    }

    return result;
}

/**************************************
purpose: yolo layer forward

Parammeter:  
             1.node0: Smallest feature map
             2.node1: Middle-size feature map
             3.node2: Largest feature map

Return: detection objects
**************************************/
std::vector<std::vector<float>> yolo_run(float* node0, float* node1, float* node2)
{
    std::vector<std::vector<float>> boxes;

    detect(boxes, node0, g_layer0_h, g_layer0_w, 2, YOLO_NET_IN_HEIGHT, YOLO_NET_IN_WIDTH);
    detect(boxes, node1, g_layer1_h, g_layer1_w, 1, YOLO_NET_IN_HEIGHT, YOLO_NET_IN_WIDTH);
    detect(boxes, node2, g_layer2_h, g_layer2_w, 0, YOLO_NET_IN_HEIGHT, YOLO_NET_IN_WIDTH);
 
    /* Apply the computation for NMS */
    boxes = applyNMS(boxes, NMS_THRESHOLD);
    // std::cout << "yolo node boxes size is:  " << boxes.size() << std::endl;

    for(size_t i = 0; i < boxes.size(); i++)
    {
        boxes[i][0] = (boxes[i][0] - boxes[i][2]/2.0f) * YOLO_NET_IN_WIDTH + 1.0f;
        boxes[i][1] = (boxes[i][1] - boxes[i][3]/2.0f) * YOLO_NET_IN_HEIGHT + 1.0f;
        boxes[i][2] = boxes[i][2] * YOLO_NET_IN_WIDTH;
        boxes[i][3] = boxes[i][3] * YOLO_NET_IN_HEIGHT;
    }

    return boxes;
}