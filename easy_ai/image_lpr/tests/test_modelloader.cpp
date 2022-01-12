//
// Created by 庾金科 on 26/12/2017.
//

#include <iostream>
#include <fstream>


#include <opencv2/dnn.hpp>


using namespace std;
using namespace cv;


struct ModelSize{
    int prototxt_size;
    int caffemodel_size;
};


struct Model{
    char *prototxtBuffer;
    char *caffemodelBuffer;
    ModelSize modelsize;
};


int main()
{



    ifstream file("model/LPR.mlz", ios::binary);
    int magic_number = 0;
    int number_of_models = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_models, sizeof(number_of_models));
    ModelSize *modelSize  = new ModelSize[number_of_models];
    file.read((char*)modelSize, sizeof(ModelSize)*number_of_models);
    Model *model = new Model[number_of_models];
    std::cout<<magic_number<<std::endl;
    std::cout<<number_of_models<<std::endl;
    for(int i = 0 ; i < number_of_models ; i ++)
    {
        model[i].prototxtBuffer = new char[modelSize[i].prototxt_size];
        model[i].caffemodelBuffer = new char[modelSize[i].caffemodel_size];
        model[i].modelsize = modelSize[i];
        file.read(model[i].prototxtBuffer, modelSize[i].prototxt_size);
        file.read(model[i].caffemodelBuffer, modelSize[i].caffemodel_size);
        cv::dnn::readNetFromCaffe(model[i].prototxtBuffer,modelSize[i].prototxt_size,model[i].caffemodelBuffer,modelSize[i].caffemodel_size);
    }

    //release memory
    for(int i = 0 ; i < number_of_models ; i ++)
    {
        delete model[i].prototxtBuffer;
        delete model[i].caffemodelBuffer;
    }
    delete model;
    delete modelSize;






    return 0 ;
}