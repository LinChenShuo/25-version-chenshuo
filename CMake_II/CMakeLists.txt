cmake_minimum_required(VERSION 3.10)

Project(Test)

find_package(OpenCV REQUIRED)

# # # # 以下内容禁止修改 # # # #
option(BUILD_A "build modules a" ON)
option(BUILD_B "build modules b" OFF)
option(BUILD_TESTS "build tests" OFF)

if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(modules)
add_subdirectory(common)
# 小提示: module2 需要用到 pthread，即 target_link_libraries(xxx pthread)
# # # # 以上内容禁止修改 # # # #

# 创建可执行文件并链接

add_executable(server server.cpp)
target_link_libraries(server PUBLIC ${OpenCV_LIBS} rmath singleton assembly1 assembly2 module1 module2)

add_executable(client client.cpp)
target_link_libraries(client PUBLIC rmath singleton assembly1 assembly2 module1 module2)

# 如果编译选项被设为on 就在相应目标中添加定义

if(BUILD_A)
    target_compile_definitions(server PRIVATE WITH_A)
endif()

if(BUILD_B)
    target_compile_definitions(server PRIVATE WITH_B)
endif()



