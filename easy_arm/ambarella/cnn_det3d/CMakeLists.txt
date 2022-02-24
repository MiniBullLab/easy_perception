PROJECT(cnn_det3d)

set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})

#include amba_arm directories
include_directories(${amba_arm_BINARY_DIR})
include_directories(${amba_arm_SOURCE_DIR})

add_definitions(-D_USE_AMBAR_)

add_executable(test_det3d)

#collect source files
file(GLOB UTILITY_SRCS
    "${amba_arm_SOURCE_DIR}/utility/*.cpp"
)

file(GLOB DRIVERS_SRCS
    "${PROJECT_SOURCE_DIR}/drivers/*.cpp"
    "${PROJECT_SOURCE_DIR}/drivers/*.S"
	)
#message(STATUS "Found ASM ${DRIVERS_SRCS}")
#set_source_files_properties(${DRIVERS_SRCS} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")

#collect source files
file(GLOB DET3D_SRCS
    "${PROJECT_SOURCE_DIR}/network/*.cpp"
	)

target_sources(test_det3d
        PRIVATE
        "${PROJECT_SOURCE_DIR}/test_det3d.cpp"
        ${DET3D_SRCS}
        ${UTILITY_SRCS}
        ${DRIVERS_SRCS}
        )

target_link_libraries(test_det3d
                      opencv_core
                      opencv_imgcodecs
                      opencv_imgproc
                      opencv_videoio
                      opencv_highgui
                      opencv_flann
                      opencv_features2d
                      opencv_calib3d
                      # opencv_dnn
                      # opencv_objdetect
                      # protobuf
                      gflags
                      glog
                      jpeg
                      png
                      z
                      nnctrl
                      vproc
                      cavalry_mem
                      smartfb
                      datatx
                      freetype
                      phase
                      cvlib
                      dl
                      rt
                      pthread
                      cnn_runtime)

