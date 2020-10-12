cmake_minimum_required(VERSION 3.0)
project(PaddleX CXX C)


option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   OFF)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
SET(LITE_DIR "" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
SET(NGRAPH_LIB "" CACHE PATH "Location of libraries")


include(cmake/yaml-cpp.cmake)

include_directories("${CMAKE_SOURCE_DIR}/")
link_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/src/ext-yaml-cpp/include")
link_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/lib")

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if (NOT DEFINED LITE_DIR OR ${LITE_DIR} STREQUAL "")
	message(FATAL_ERROR "please set LITE_DIR with -LITE_DIR=/path/influence_engine")
endif()

if (NOT DEFINED OPENCV_DIR OR ${OPENCV_DIR} STREQUAL "")
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

if (NOT DEFINED GFLAGS_DIR OR ${GFLAGS_DIR} STREQUAL "")
    message(FATAL_ERROR "please set GFLAGS_DIR with -DGFLAGS_DIR=/path/gflags")
endif()





link_directories("${LITE_DIR}/lib")
include_directories("${LITE_DIR}/include")



link_directories("${GFLAGS_DIR}/lib")
include_directories("${GFLAGS_DIR}/include")




if (WIN32)
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build/ NO_DEFAULT_PATH)
  unset(OpenCV_DIR CACHE)
else ()
	find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
endif ()

include_directories(${OpenCV_INCLUDE_DIRS})

if (WIN32)
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
    if (WITH_STATIC_LIB)
        safe_set_static_flag()
        add_definitions(-DSTATIC_LIB)
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat-abi=hard -mfpu=neon-vfpv4 -g -o2 -fopenmp -std=c++11")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
endif()


if(WITH_STATIC_LIB)
	set(DEPS ${LITE_DIR}/lib/libpaddle_full_api_shared${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
	set(DEPS ${LITE_DIR}/lib/libpaddle_full_api_shared${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

if (NOT WIN32)
    set(DEPS ${DEPS}
        gflags yaml-cpp
        )
else()
    set(DEPS ${DEPS}
        gflags_static libprotobuf zlibstatic xxhash libyaml-cppmt)
    set(DEPS ${DEPS} libcmt shlwapi)
endif(NOT WIN32)


if (NOT WIN32)
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

set(DEPS ${DEPS} ${OpenCV_LIBS})
add_executable(classifier demo/classifier.cpp src/transforms.cpp src/paddlex.cpp)
ADD_DEPENDENCIES(classifier ext-yaml-cpp)
target_link_libraries(classifier ${DEPS})


add_executable(segmenter demo/segmenter.cpp src/transforms.cpp src/paddlex.cpp src/visualize.cpp)
ADD_DEPENDENCIES(segmenter ext-yaml-cpp)
target_link_libraries(segmenter ${DEPS})

add_executable(detector demo/detector.cpp src/transforms.cpp src/paddlex.cpp src/visualize.cpp)
ADD_DEPENDENCIES(detector ext-yaml-cpp)
target_link_libraries(detector ${DEPS})
