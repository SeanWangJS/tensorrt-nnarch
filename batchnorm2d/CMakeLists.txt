cmake_minimum_required(VERSION 3.20)

project(batchnorm2d)

set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

if(TensorRT_ROOT)
    message(STATUS "TensorRT_ROOT: ${TensorRT_ROOT}")
    if(NOT TensorRT_INCLUDE_DIR)
        set(TensorRT_INCLUDE_DIR ${TensorRT_ROOT}/include)
    endif()
    if(NOT TensorRT_LIB_DIR)
        set(TensorRT_LIB_DIR ${TensorRT_ROOT}/lib)
    endif()
endif()

file(GLOB_RECURSE SOURCES src/*.cu)

add_executable(batchnorm2d ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC ${TensorRT_INCLUDE_DIR}
                                                    ${PROJECT_SOURCE_DIR}/../include)

target_link_libraries(${PROJECT_NAME} 
                        ${TensorRT_LIB_DIR}/libnvinfer.so)