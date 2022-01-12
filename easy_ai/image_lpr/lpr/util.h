#ifndef UTIL_H
#define UTIL_H
#include <opencv2/opencv.hpp>

namespace util{

    template <class T> void swap ( T& a, T& b )
    {
        T c(a); a=b; b=c;
    }

    template <class T> T min(T& a,T& b )
    {
        return a>b?b:a;

    }
    enum Color { BLUE, YELLOW, WHITE, GREEN,UNKNOWN };

    cv::Mat colorMatch(const cv::Mat &src, cv::Mat &match, const Color r,
                   const bool adaptive_minsv) {

        // if use adaptive_minsv
        // min value of s and v is adaptive to h
        const float max_sv = 255;
        const float minref_sv = 64;

        const float minabs_sv = 95; //95;

        // H range of blue

        const int min_blue = 100;  // 100
        const int max_blue = 140;  // 140

        // H range of yellow

        const int min_yellow = 15;  // 15
        const int max_yellow = 40;  // 40

        // H range of yellow
        const int min_green = 35;  // 35
        const int max_green = 77;  // 77

        // H range of white

        const int min_white = 0;   // 15
        const int max_white = 30;  // 40

        cv::Mat src_hsv;

        // convert to HSV space
        cvtColor(src, src_hsv, CV_BGR2HSV);

        std::vector<cv::Mat> hsvSplit;
        split(src_hsv, hsvSplit);
        equalizeHist(hsvSplit[2], hsvSplit[2]);
        merge(hsvSplit, src_hsv);

        // match to find the color

        int min_h = 0;
        int max_h = 0;
        switch (r) {
            case BLUE:
                min_h = min_blue;
                max_h = max_blue;
                break;
            case YELLOW:
                min_h = min_yellow;
                max_h = max_yellow;
                break;
            case WHITE:
                min_h = min_white;
                max_h = max_white;
                break;
            case GREEN:
                min_h = min_green;
                max_h = max_green;
                break;
            default:
                // Color::UNKNOWN
                break;
        }

        float diff_h = float((max_h - min_h) / 2);
        float avg_h = min_h + diff_h;

        int channels = src_hsv.channels();
        int nRows = src_hsv.rows;

        // consider multi channel image
        int nCols = src_hsv.cols * channels;
        if (src_hsv.isContinuous()) {
            nCols *= nRows;
            nRows = 1;
        }

        int i, j;
        uchar* p;
        float s_all = 0;
        float v_all = 0;
        float count = 0;
        for (i = 0; i < nRows; ++i) {
            p = src_hsv.ptr<uchar>(i);
            for (j = 0; j < nCols; j += 3) {
                int H = int(p[j]);      // 0-180
                int S = int(p[j + 1]);  // 0-255
                int V = int(p[j + 2]);  // 0-255

                s_all += S;
                v_all += V;
                count++;

                bool colorMatched = false;

                if (H > min_h && H < max_h) {
                    float Hdiff = 0;
                    if (H > avg_h)
                        Hdiff = H - avg_h;
                    else
                        Hdiff = avg_h - H;

                    float Hdiff_p = float(Hdiff) / diff_h;

                    float min_sv = 0;
                    if (true == adaptive_minsv)
                        min_sv =
                                minref_sv -
                                minref_sv / 2 *
                                (1
                                 - Hdiff_p);  // inref_sv - minref_sv / 2 * (1 - Hdiff_p)
                    else
                        min_sv = minabs_sv;  // add

                    if ((S > min_sv && S < max_sv) && (V > min_sv && V < max_sv))
                        colorMatched = true;
                }

                if (colorMatched == true) {
                    p[j] = 0;
                    p[j + 1] = 0;
                    p[j + 2] = 255;
                }
                else {
                    p[j] = 0;
                    p[j + 1] = 0;
                    p[j + 2] = 0;
                }
            }
        }
        cv::Mat src_grey;
        std::vector<cv::Mat> hsvSplit_done;
        split(src_hsv, hsvSplit_done);
        src_grey = hsvSplit_done[2];
        match = src_grey;
        return src_grey;
    }

     bool plateColorJudge(const cv::Mat &src, const Color r, const bool adaptive_minsv,
                       float &percent) {

    const float thresh = 0.45f;

    cv::Mat src_gray;
    colorMatch(src, src_gray, r, adaptive_minsv);

    percent =
        float(countNonZero(src_gray)) / float(src_gray.rows * src_gray.cols);
    // cout << "percent:" << percent << endl;

    if (percent > thresh)
      return true;
    else
      return false;
  }

  Color getPlateType(const cv::Mat &src, const bool adaptive_minsv) {
    float max_percent = 0;
    Color max_color = UNKNOWN;

    float blue_percent = 0;
    float yellow_percent = 0;
    float white_percent = 0;

    if (plateColorJudge(src, BLUE, adaptive_minsv, blue_percent) == true) {
      // cout << "BLUE" << endl;
      return BLUE;
    } else if (plateColorJudge(src, YELLOW, adaptive_minsv, yellow_percent) ==
               true) {
      // cout << "YELLOW" << endl;
      return YELLOW;
    } else if (plateColorJudge(src, WHITE, adaptive_minsv, white_percent) ==
               true) {
      // cout << "WHITE" << endl;
		return YELLOW;
    }

    else if (plateColorJudge(src, GREEN, adaptive_minsv, white_percent) ==
             true) {
        // cout << "WHITE" << endl;
		return GREEN;
    }
    else {
      //std::cout << "OTHER" << std::endl;
      /*max_percent = blue_percent > yellow_percent ? blue_percent : yellow_percent;
      max_color = blue_percent > yellow_percent ? BLUE : YELLOW;
      max_color = max_percent > white_percent ? max_color : WHITE;*/
      // always return blue
      return BLUE;
    }
  }

    cv::Mat cropFromImage(const cv::Mat &image,cv::Rect rect){
        int w = image.cols-1;
        int h = image.rows-1;
        rect.x = std::max(rect.x,0);
        rect.y = std::max(rect.y,0);
        rect.height = std::min(rect.height,h-rect.y);
        rect.width = std::min(rect.width,w-rect.x);
        cv::Mat temp(rect.size(), image.type());
        cv::Mat cropped;
        temp = image(rect);
        temp.copyTo(cropped);
        return cropped;

    }

    cv::Mat cropBox2dFromImage(const cv::Mat &image,cv::RotatedRect rect)
    {
        cv::Mat M, rotated, cropped;
        float angle = rect.angle;
        cv::Size rect_size(rect.size.width,rect.size.height);
        if (rect.angle < -45.) {
            angle += 90.0;
            swap(rect_size.width, rect_size.height);
        }
        M = cv::getRotationMatrix2D(rect.center, angle, 1.0);
        cv::warpAffine(image, rotated, M, image.size(), cv::INTER_CUBIC);
        cv::getRectSubPix(rotated, rect_size, rect.center, cropped);
        return cropped;
    }

    cv::Mat calcHist(const cv::Mat &image)
    {
        cv::Mat hsv;
        std::vector<cv::Mat> hsv_planes;
        cv::cvtColor(image,hsv,cv::COLOR_BGR2HSV);
        cv::split(hsv,hsv_planes);
        cv::Mat hist;
        int histSize = 256;
        float range[] = {0,255};
        const float* histRange = {range};

        cv::calcHist( &hsv_planes[0], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange,true, true);
        return hist;

    }
    float computeSimilir(const cv::Mat &A,const cv::Mat &B)
    {
        cv::Mat histA,histB;
        histA = calcHist(A);
        histB = calcHist(B);
        return cv::compareHist(histA,histB,CV_COMP_CORREL);

    }





}//namespace util
#endif
