
#include "CNNRecognizer.h"

namespace pr{
    CNNRecognizer::CNNRecognizer(std::string prototxt,std::string caffemodel){
        net = cv::dnn::readNetFromCaffe(prototxt, caffemodel);
    }

    CNNRecognizer::CNNRecognizer(ModelLoader *modeloader){
        Model* model= modeloader->readModel(0);//
        net = cv::dnn::readNetFromCaffe(model->prototxtBuffer,model->modelsize.prototxt_size,model->caffemodelBuffer,model->modelsize.caffemodel_size);

    }



    label CNNRecognizer::recognizeCharacter(cv::Mat charImage){
        if(charImage.channels()== 3)
            cv::cvtColor(charImage,charImage,cv::COLOR_BGR2GRAY);
        cv::equalizeHist(charImage,charImage);

        cv::Mat inputBlob = cv::dnn::blobFromImage(charImage, 1/255.0, cv::Size(CHAR_INPUT_W,CHAR_INPUT_H), cv::Scalar(0,0,0),false);
        net.setInput(inputBlob,"data");
        return net.forward();
    }
}