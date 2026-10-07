cmake_minimum_required(VERSION 3.16)

project(task2_armor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs calib3d video)

add_executable(task2_armor
    src/main.cpp
    src/armor_detector.cpp
)

target_include_directories(task2_armor PRIVATE include)
target_link_libraries(task2_armor PRIVATE
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
)

if(MSVC)
    target_compile_options(task2_armor PRIVATE /W4 /permissive-)
else()
    target_compile_options(task2_armor PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(optional_armor
    src/optional_main.cpp
    src/optional_armor.cpp
)
target_include_directories(optional_armor PRIVATE include)
target_link_libraries(optional_armor PRIVATE
    opencv_core
    opencv_imgproc
    opencv_videoio
    opencv_imgcodecs
    opencv_calib3d
    opencv_video
)
if(MSVC)
    target_compile_options(optional_armor PRIVATE /W4 /permissive-)
else()
    target_compile_options(optional_armor PRIVATE -Wall -Wextra -Wpedantic)
endif()

