cmake_minimum_required(VERSION 3.10)

project(Main)

# SET(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

add_library(mylib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mylib.cpp)
target_include_directories(mylib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mylib PUBLIC ${OpenCV_LIBS})

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} mylib)


