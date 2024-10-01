# 搜索包
find_package(OpenCV REQUIRED)
# 添加一个静态库目标
add_library(M2 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/M2.cpp)
# 静态库链接头文件和库文件
target_include_directories(M2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(M2 PUBLIC A1 A2 kalman)