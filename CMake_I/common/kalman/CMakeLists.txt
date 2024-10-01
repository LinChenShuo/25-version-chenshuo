# 搜索包
find_package(OpenCV REQUIRED)
# 添加一个接口库目标
add_library(kalman INTERFACE)
# 接口库链接头文件和库文件
target_include_directories(kalman INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kalman INTERFACE ${OpenCV_LIBS})
