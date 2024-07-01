cmake_minimum_required(VERSION 3.20)

project(batchnorm2d)

set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

file(GLOB_RECURSE SOURCES src/*.cu)

add_executable(batchnorm2d ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC /usr/local/tensorrt/include
                                                    ${PROJECT_SOURCE_DIR}/../include)

target_link_libraries(${PROJECT_NAME} 
                        /usr/local/tensorrt/lib/libnvinfer.so)