project(test_imgproc)

#include easy_ai directories
include_directories(${easy_ai_BINARY_DIR})
include_directories(${easy_ai_SOURCE_DIR})

#collect source files
file(GLOB TEST_IMGPROC_SRCS 
    "${PROJECT_SOURCE_DIR}/*.cpp"
)

add_executable(test_imgproc ${TEST_IMGPROC_SRCS})
target_link_libraries(test_imgproc imgproc_c opencv_core opencv_videoio opencv_imgcodecs opencv_imgproc opencv_highgui)
