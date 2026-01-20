cmake_minimum_required(VERSION 3.10)
project(FlashAttention_Demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 指向正确的文件夹名称 src_c
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src_c)

set(SOURCES 
    ${SRC_DIR}/main.cpp 
    ${SRC_DIR}/tool.cpp
)

add_executable(flash_atten_demo ${SOURCES})
target_include_directories(flash_atten_demo PRIVATE ${SRC_DIR})
target_link_libraries(flash_atten_demo m)