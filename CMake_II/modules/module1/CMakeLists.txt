find_package(OpenCV REQUIRED)
# 找路径下的所有文件并输出为变量列表
file(GLOB src_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/*")

foreach(m ${src_dir})
    # 如果是目录
    if(IS_DIRECTORY ${m})
        # 搜寻该目录下的源文件并添加到一个列表中，方便一次性加载
        aux_source_directory(${m} src_cpp)
        list(APPEND src ${src_cpp})
    endif()
endforeach(m ${src_dir})

add_library(module1 ${src})
target_include_directories(module1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(module1 PUBLIC ${OPENCV_LIBS} assembly2)

message(STATUS "module1 ok")