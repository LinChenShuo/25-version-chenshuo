# 添加一个静态库目标
add_library(M1 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/M1.cpp)
# 静态库链接头文件和库文件
target_include_directories(M1 PUBLIC ${CMAKE_SOURCE_DIR}/modules/M1/include)
target_link_libraries(M1 PUBLIC A1)

