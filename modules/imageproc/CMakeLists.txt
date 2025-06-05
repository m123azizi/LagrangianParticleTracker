project("module-imageproc")
message(STATUS "Adding module imageproc")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src"
                        "${CMAKE_CURRENT_BINARY_DIR}")
        
include_directories("${CMAKE_SOURCE_DIR}/modules/core/include")

message(STATUS "   depends on module: core")

file(GLOB lib_int_hdrs "src/*.h*")
file(GLOB lib_cu_srcs "src/*.cu")
file(GLOB lib_cpp_srcs "src/*.cpp")
source_group("Src" FILES ${lib_cpp_srcs} ${lib_cu_srcs} ${lib_int_hdrs})

file(GLOB lib_hdrs "include/*.h*")
source_group("Include" FILES ${lib_hdrs})

message(STATIC " IMAGE PROC LIBRARY IS: ${PROJECT_NAME}")

cuda_add_library(${PROJECT_NAME} ${lib_cu_srcs} ${lib_cpp_srcs} ${lib_hdrs} ${lib_int_hdrs})
    
target_link_libraries(${PROJECT_NAME} module-core ${CUDA_LIBRARIES})
add_dependencies(${PROJECT_NAME} module-core)

if(Boost_FOUND)
	target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()   


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})