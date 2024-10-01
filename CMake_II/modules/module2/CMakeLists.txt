find_package(OpenCV REQUIRED)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src src)
add_library(module2 ${src})
target_include_directories(module2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

message(STATUS "module2 ok")