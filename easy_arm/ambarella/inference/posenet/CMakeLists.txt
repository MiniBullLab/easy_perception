PROJECT(posenet)

#include amba_arm directories
include_directories(${amba_arm_BINARY_DIR})
include_directories(${amba_arm_SOURCE_DIR})

#collect source files
file(GLOB COMMON_SRCS
    "${amba_arm_SOURCE_DIR}/inference/common/*.cpp"
)

add_executable(posenet_test posenet_test.cpp openpose_postprocess.cpp ${COMMON_SRCS})
target_link_libraries(posenet_test 
                      opencv_core
                      opencv_imgcodecs
                      opencv_imgproc
                      opencv_highgui
                      jpeg
                      png
                      z
                      nnctrl
                      vproc
                      cavalry_mem)