cmake_minimum_required(VERSION 3.6)
project(image_lpr)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(..)


set(SRC_DETECTION lpr/PlateDetection.cpp lpr/util.h lpr/PlateDetection.h)

set(SRC_FINEMAPPING lpr/FineMapping.cpp )

set(SRC_FASTDESKEW lpr/FastDeskew.cpp )

set(SRC_SEGMENTATION lpr/PlateSegmentation.cpp )

set(SRC_RECOGNIZE lpr/Recognizer.cpp lpr/CNNRecognizer.cpp)

set(SRC_PIPLINE lpr/Pipeline.cpp   )

set(SRC_SEGMENTATIONFREE lpr/SegmentationFreeRecognizer.cpp )

set(SRC_MODELLOADER lpr/ModelLoader.cpp )

#set(SRC_SINGLESHOT lpr/SingleShotDetector.cpp)




#TEST_PIPELINE
add_executable(TEST_PIPLINE ${SRC_MODELLOADER} ${SRC_FINEMAPPING} ${SRC_FASTDESKEW} ${SRC_SEGMENTATION} ${SRC_RECOGNIZE} ${SRC_SEGMENTATIONFREE} ${SRC_DETECTION} ${SRC_PIPLINE}  tests/test_pipeline.cpp)
target_link_libraries(TEST_PIPLINE ${OpenCV_LIBS} )


