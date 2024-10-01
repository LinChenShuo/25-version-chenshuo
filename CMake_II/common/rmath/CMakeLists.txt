find_package(OpenCV REQUIRED)
add_library(rmath ${CMAKE_CURRENT_SOURCE_DIR}/src/rmath.cpp)
target_include_directories(rmath PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rmath PUBLIC ${OpenCV_LIBS})

message(STATUS "rmath ok")
