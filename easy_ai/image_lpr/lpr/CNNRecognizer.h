
#ifndef SWIFTPR_CNNRECOGNIZER_H
#define SWIFTPR_CNNRECOGNIZER_H

#include "Recognizer.h"
#include "ModelLoader.h"

namespace pr{
    class CNNRecognizer: public GeneralRecognizer{
    public:
        const int CHAR_INPUT_W = 14;
        const int CHAR_INPUT_H = 30;

        CNNRecognizer(std::string prototxt,std::string caffemodel);
        CNNRecognizer(ModelLoader* modeloader);
        label recognizeCharacter(cv::Mat character);
    private:
        cv::dnn::Net net;

    };

}

#endif //SWIFTPR_CNNRECOGNIZER_H
