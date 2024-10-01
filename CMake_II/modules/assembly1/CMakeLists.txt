add_library(assembly1 ${CMAKE_CURRENT_SOURCE_DIR}/src/assembly1.cpp)
target_include_directories(assembly1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加要测试用的可执行文件
add_executable(assembly1_test ${CMAKE_CURRENT_SOURCE_DIR}/test/assembly1_test.cpp)
target_link_libraries(assembly1_test PUBLIC assembly1)

#添加测试项
add_test(NAME test COMMAND assembly1_test)
message(STATUS "assembly1 ok")