# 搜索包
find_package(OpenCV REQUIRED)
# 添加一个静态库目标
add_library(math STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/Math.cpp)
# 静态库链接头文件和库文件
target_include_directories(math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#find_package后要link一下library
target_link_libraries(math ${OpenCV_LIBS})
