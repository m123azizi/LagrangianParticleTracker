# Main CMakeLists file

cmake_minimum_required (VERSION 3.4)
project (LPT)
if (POLICY CMP0146) 
	cmake_policy(SET CMP0146 OLD) 
endif()
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/scripts/cmake/")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
  
include( "./scripts/cmake/LptCmakeMacros.cmake" REQUIRED)  
find_package( OpenCV 4.8.0 REQUIRED )

find_package( VTK 8.2.0 REQUIRED)
find_package( YamlCPP REQUIRED )
find_package( Qt5 )
set( QT_USE_QTOPENGL TRUE )

find_package( CUDA 11.8 )
find_package(Torch REQUIRED)

message(STATUS "OpenCV include: " ${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV library: " ${OpenCV_LIB_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})

link_directories( ${VTK_LIBRARY_DIRS} ) 
include_directories( ${VTK_INCLUDE_DIRS} )
message(STATUS "VTK include: " ${VTK_INCLUDE_DIRS})
message(STATUS "VTK lib: " ${VTK_LIBRARY_DIRS})

if (CUDA_FOUND)
	link_directories( ${CUDA_LIBRARY_DIRS} )
	include_directories( ${CUDA_INCLUDE_DIRS} )
	add_definitions( -DUSE_CUDA )
	message(STATUS "CUDA include: " ${CUDA_INCLUDE_DIRS})
	message(STATUS "CUDA toolkit: " ${CUDA_TOOLKIT_ROOT_DIR})
endif()

set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
set(Boost_NO_BOOST_CMAKE         ON)
set(BOOST_ROOT C:/mahdi/VPTV_VS2022/packages/boost_1_83_0)

set(Boost_ADDITIONAL_VERSIONS "1.83" "1.83.0")


find_package( Boost 1.83.0 COMPONENTS thread chrono random date_time system REQUIRED )
set(Boost_INCLUDE_DIRS ${BOOST_ROOT})
set(Boost_LIBRARY_DIRS ${BOOST_ROOT}/stage/lib)
set(BOOST_LIBRARYDIR ${BOOST_ROOT}/stage/lib)
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}") 
message(STATUS "Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}") 
message(STATUS "Boost_LIBRARYDIR: ${BOOST_LIBRARYDIR}") 
message(STATUS "BEFORE NPCAMERA")
if (WIN32)
    message(STATUS "IN NPCAMERA")
    find_package( NP_CAMERASDK )
	#add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
    include_directories($ENV{MSMPI_INC})
endif()

include_directories(${Boost_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

message(STATUS "Starting LPT library build:")
add_subdirectory("${CMAKE_SOURCE_DIR}/modules")
add_subdirectory("${CMAKE_SOURCE_DIR}/apps")
add_subdirectory("${CMAKE_SOURCE_DIR}/test")