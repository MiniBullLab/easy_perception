PROJECT(rec_text)

#include amba_arm directories
include_directories(${amba_arm_BINARY_DIR})
include_directories(${amba_arm_SOURCE_DIR})

#collect source files
file(GLOB COMMON_SRCS
    "${amba_arm_SOURCE_DIR}/inference/common/*.cpp"
)

add_library(textnet SHARED textnet.cpp ${COMMON_SRCS})

add_executable(textnet_test textnet_test.cpp)
target_link_libraries(textnet_test
                      textnet 
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