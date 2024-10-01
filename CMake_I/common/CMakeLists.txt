#执行ls命令之后的返回值输出为变量
execute_process(COMMAND ls ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE TEST_MODULE_COMMON_ST)
#上述返回值是用\n分隔，改成;分隔（cmake里是用空格或者分号分隔变量）
string(REPLACE "\n" ";" TEST_MODULE_COMMON_ST "${TEST_MODULE_COMMON_ST}")
#对于列表中的每一项
foreach(m ${TEST_MODULE_COMMON_ST})
    #如果是目录
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${m})
        #搜索子目录下的CMakeLists.txt
        add_subdirectory(${m})
    endif()
endforeach()
