# 搜索源文件
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src src)
# 添加一个静态库目标
add_library(A1 STATIC ${src})
# 静态库链接头文件
target_include_directories(A1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
