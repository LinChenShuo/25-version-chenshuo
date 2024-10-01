# 在此处进行 CMake 设置（最小版本号，项目名）
cmake_minimum_required(VERSION 3.10)

# 项目名配置
project(Main)

# 搜索包
find_package(OpenCV REQUIRED)

####### 以下禁止修改 #######
add_subdirectory(common)   #
add_subdirectory(modules)  #
####### 以上禁止修改 #######

# 在此处生成可执行程序
add_executable(main main.cpp)
# 在此处链接库
target_link_libraries(main PUBLIC math M1 M2)
