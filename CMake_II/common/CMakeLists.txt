# 找路径下的所有文件并输出为变量列表
file(GLOB COMMON_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/*")

# 对于该列表里的每个路径
foreach(m ${COMMON_DIRS})
    # 如果是目录
    if(IS_DIRECTORY ${m})
        # 查询对应的子目录并执行里面的CMakelists.txt
        add_subdirectory(${m})
    endif()
endforeach()