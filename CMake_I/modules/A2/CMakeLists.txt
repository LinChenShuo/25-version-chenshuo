# 添加一个静态库目标
add_library(A2 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/A2.cpp)
# 静态库链接头文件和库文件
target_include_directories(A2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 补充头文件满足size_t的定义
target_compile_options(A2 PUBLIC -include cstddef)