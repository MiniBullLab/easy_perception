PROJECT(cnn_lpr)

#include amba_arm directories
include_directories(${amba_arm_BINARY_DIR})
include_directories(${amba_arm_SOURCE_DIR})

add_definitions(-D_USE_AMBAR_)

#collect source files
file(GLOB UTILITY_SRCS
    "${amba_arm_SOURCE_DIR}/utility/*.cpp"
)

#collect source files
file(GLOB LPR_SRCS
    "${PROJECT_SOURCE_DIR}/common/*.cpp"
    "${PROJECT_SOURCE_DIR}/image/*.cpp"
    "${PROJECT_SOURCE_DIR}/led/*.cpp"
    "${PROJECT_SOURCE_DIR}/net_process/*.cpp"
    "${PROJECT_SOURCE_DIR}/lpr/*.cpp"
	"${PROJECT_SOURCE_DIR}/lpr/*.c"
    "${PROJECT_SOURCE_DIR}/lpr_third_party/FineMapping.cpp"
    "${PROJECT_SOURCE_DIR}/lpr_third_party/FastDeskew.cpp"
    "${PROJECT_SOURCE_DIR}/lpr_third_party/Pipeline.cpp"
    "${PROJECT_SOURCE_DIR}/tof/*.cpp"
	)

add_executable(test_lpr test_lpr.cpp ${LPR_SRCS} ${UTILITY_SRCS})
target_link_libraries(test_lpr
                      opencv_core
                      opencv_imgcodecs
                      opencv_imgproc
                      opencv_videoio
                      # opencv_highgui
                      # opencv_dnn
                      # opencv_flann
                      # opencv_calib3d
                      # opencv_objdetect
                      # protobuf
                      gflags
                      glog
                      tbb
                      jpeg
                      png
                      z
                      _data_process
                      eazyai_opencv
                      nnctrl
                      vproc
                      cavalry_mem
                      smartfb
                      textinsert_v2
                      bz2
                      freetype
                      phase
                      cvlib
                      dl
                      rt
                      pthread)

