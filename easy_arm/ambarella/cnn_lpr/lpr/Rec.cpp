#include "Rec.h"
#include <fstream>
#include <numeric>
#include <iostream>
#include "rec.mem.h"
#include "rec.id.h"

const std::vector<std::string> CH_PLATE_CODE{"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
									"琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新","港","学","使","警","澳","挂","民","航","领","应","急", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
									"B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X","Y", "Z"};


RecNet::~RecNet() {
	net.clear();
}

bool RecNet::initModel(const std::string& pathStr) {
	int ret_param = net.load_param(__rec_op_param_bin);
	int ret_bin = net.load_model((pathStr).c_str());

	if (ret_param = 0 || ret_bin != 0) {
		printf("RecNet load param(%d), model(%d)\n", ret_param, ret_bin);
		return false;
	}
	
	return true;
}

cv::Mat RecNet::cropImageROI(const cv::Mat &srcImage, const std::vector<cv::Point> &polygon)
{
    cv::Scalar borderValue;
    if(srcImage.channels() == 1)
    {
        borderValue = cv::Scalar(0);
    }
    else
    {
        borderValue = cv::Scalar(0, 0, 0);
    }
    cv::Point2f srcpoint[4];//存放变换前四顶点
    cv::Point2f dstpoint[4];//存放变换后四顶点
    cv::RotatedRect rect = cv::minAreaRect(polygon);
    float width = rect.size.width;
    float height = rect.size.height;
    rect.points(srcpoint);//获取最小外接矩形四顶点坐标
    dstpoint[0]= cv::Point2f(0, height);
    dstpoint[1] = cv::Point2f(0, 0);
    dstpoint[2] = cv::Point2f(width, 0);
    dstpoint[3] = cv::Point2f(width, height);
    cv::Mat M = cv::getPerspectiveTransform(srcpoint, dstpoint);
    cv::Mat result = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
    cv::warpPerspective(srcImage, result, M, result.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValue);
    std::cout << "width:" << result.cols << " height:" << result.rows << std::endl;
    if((result.rows * 1.0f / result.cols) >= 2.0f)
    {
        // cv::Point2f center;
        // center.x = float(result.cols / 2.0);
        // center.y = float(result.rows / 2.0);
        // int length = cv::sqrt(result.cols * result.cols + result.rows * result.rows);
        // cv::Mat M = cv::getRotationMatrix2D(center, -90, 1);
        // cv::warpAffine(src, src_rotate, M, Size(length, length), 1, 0, Scalar(0, 0, 0));
        cv::Mat temp;
        cv::transpose(result, temp);
        cv::flip(temp, result, 0);
    }
    return result;
}

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
	return std::distance(first, std::max_element(first, last));
}

TextLine RecNet::scoreToTextLine(const std::vector<float>& outputData, int h, int w) {
	int keySize = CH_PLATE_CODE.size();
	std::string strRes;
	std::vector<float> scores;
	int lastIndex = 0;
	int maxIndex;
	float maxValue;

	for (int i = 0; i < h; i++) {
		maxIndex = 0;
		maxValue = -1000.f;
		maxIndex = int(argmax(outputData.begin() + i * w, outputData.begin() + i * w + w));
		maxValue = float(*std::max_element(outputData.begin() + i * w, outputData.begin() + i * w + w));
		if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex))) {
			scores.emplace_back(maxValue);
			strRes.append(CH_PLATE_CODE[maxIndex - 1]);
		}
		lastIndex = maxIndex;
	}
	return { strRes, scores };
}

static void resize_image(const cv::Mat input, cv::Mat& out){
	int imgH = 48, imgW = 160;
	float max_wh_ratio = (float)imgW*1.0 / imgH;
	int h = input.rows;
	int w = input.cols;
	float ratio = w * 1.0 / h;
	max_wh_ratio = std::max(max_wh_ratio, ratio);

	imgW = int(32 * max_wh_ratio);
	int resize_w = 0, resize_h = 0;
	if (std::ceil(imgH * ratio) > imgW){
		resize_w = imgW;
	}
	else{
		resize_w = int(std::ceil(imgH * ratio));
	}
	cv::Mat resized_img;
	cv::resize(input, resized_img, cv::Size(resize_w, imgH), 0, 0, 1);
	cv::Mat padding_img = cv::Mat::zeros(cv::Size(imgW,imgH),CV_8UC3);
	resized_img.copyTo(padding_img(cv::Rect(0, 0, resized_img.cols, resized_img.rows)));
	padding_img.copyTo(out);

}

TextLine RecNet::getTextLine(const cv::Mat & src) {
	cv::Mat srcResize;
	resize_image(src, srcResize);

	ncnn::Mat input = ncnn::Mat::from_pixels(
		srcResize.data, ncnn::Mat::PIXEL_BGR,
		srcResize.cols, srcResize.rows);

	input.substract_mean_normalize(meanValues, normValues);

	ncnn::Extractor extractor = net.create_extractor();
	extractor.set_num_threads(4);
	extractor.input(__rec_op_param_id::BLOB_input, input);

	ncnn::Mat out;
	extractor.extract(__rec_op_param_id::BLOB_output, out);
	float* floatArray = (float*)out.data;
	std::vector<float> outputData(floatArray, floatArray + out.h * out.w);
	return scoreToTextLine(outputData, out.h, out.w);
}

std::vector<TextLine> RecNet::getTextLines(std::vector<cv::Mat> & partImg) {
	int size = partImg.size();
	std::vector<TextLine> textLines(size);
	for (int i = 0; i < size; ++i) 
	{
		TextLine textLine = getTextLine(partImg[i]);
		textLines[i] = textLine;
	}
	return textLines;
}