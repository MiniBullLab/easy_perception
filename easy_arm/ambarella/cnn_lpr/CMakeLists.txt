PROJECT(cnn_lpr)

#include amba_arm directories
include_directories(${amba_arm_BINARY_DIR})
include_directories(${amba_arm_SOURCE_DIR})

#collect source files
file(GLOB LPR_SRCS
    "${PROJECT_SOURCE_DIR}/ssd/*.c"
    "${PROJECT_SOURCE_DIR}/lpr/*.cpp"
	"${PROJECT_SOURCE_DIR}/lpr/*.c"
    "${PROJECT_SOURCE_DIR}/lpr_third_party/*.cpp"
	)

add_executable(test_lpr test_lpr.cpp ${LPR_SRCS})
target_link_libraries(test_lpr
                      opencv_core
                      opencv_imgcodecs
                      opencv_imgproc
                      opencv_highgui
                      opencv_dnn
                      opencv_ccalib
                      opencv_objdetect
                      protobuf
                      tbb
                      jpeg
                      png
                      z
                      _data_process
                      eazyai 
                      nnctrl
                      vproc
                      cavalry_mem
                      smartfb
                      textinsert_v2
                      freetype)

