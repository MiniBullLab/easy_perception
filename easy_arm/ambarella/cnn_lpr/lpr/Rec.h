#ifndef __REC_H__
#define __REC_H__

#include "ncnn/net.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

struct TextLine {
	std::string text;
	std::vector<float> charScores;
};

class RecNet {
public:

	~RecNet();

	bool initModel(const std::string& pathStr);

	cv::Mat cropImageROI(const cv::Mat &srcImage, const std::vector<cv::Point2f> &polygon);
	
	std::vector<TextLine> getTextLines(std::vector<cv::Mat>& partImg);

	TextLine getTextLine(const cv::Mat& src);

private:
	ncnn::Net net;

	const float meanValues[3] = { 127.5, 127.5, 127.5 };
	const float normValues[3] = { 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };
	
	std::vector<std::string> keys;

	TextLine scoreToTextLine(const std::vector<float>& outputData, int h, int w);
};


#endif //__RecNet_H__
