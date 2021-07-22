cmake_minimum_required(VERSION 3.0)
project(PaddleX CXX C)


option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   OFF)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
SET(OPENVINO_DIR "" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
SET(GFLAGS_DIR "" CACHE PATH "Location of libraries")
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

if(NOT WIN32)
    if (NOT DEFINED ARCH OR ${ARCH} STREQUAL "")
        message(FATAL_ERROR "please set ARCH with -DARCH=x86 OR armv7")
    endif()
endif()

if (NOT DEFINED OPENVINO_DIR OR ${OPENVINO_DIR} STREQUAL "")
    message(FATAL_ERROR "please set OPENVINO_DIR with -DOPENVINO_DIR=/path/influence_engine")
endif()

if (NOT DEFINED OPENCV_DIR OR ${OPENCV_DIR} STREQUAL "")
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

if (NOT DEFINED GFLAGS_DIR OR ${GFLAGS_DIR} STREQUAL "")
    message(FATAL_ERROR "please set GFLAGS_DIR with -DGFLAGS_DIR=/path/gflags")
endif()


if (NOT DEFINED NGRAPH_LIB OR ${NGRAPH_LIB} STREQUAL "")
    message(FATAL_ERROR "please set NGRAPH_DIR with -DNGRAPH_DIR=/path/ngraph")
endif()

include_directories("${OPENVINO_DIR}")
include_directories("${OPENVINO_DIR}/include")
include_directories("${OPENVINO_DIR}/external/tbb/include/tbb")
link_directories("${OPENVINO_DIR}/lib")
link_directories("${OPENVINO_DIR}/external/tbb/lib")
if(WIN32)
    link_directories("${OPENVINO_DIR}/lib/intel64/Release")
    link_directories("${OPENVINO_DIR}/bin/intel64/Release")
endif()



link_directories("${GFLAGS_DIR}/lib")
include_directories("${GFLAGS_DIR}/include")

link_directories("${NGRAPH_LIB}")
link_directories("${NGRAPH_LIB}/lib")

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o2 -fopenmp -std=c++11")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
endif()

if(WIN32)
    set(DEPS ${OPENVINO_DIR}/lib/intel64/Release/inference_engine${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/Release/inference_engine_legacy${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
    if (${ARCH} STREQUAL "armv7")
        if(WITH_STATIC_LIB)
            set(DEPS ${OPENVINO_DIR}/lib/armv7l/libinference_engine${CMAKE_STATIC_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/armv7l/libinference_engine_legacy${CMAKE_STATIC_LIBRARY_SUFFIX})
        else()
            set(DEPS ${OPENVINO_DIR}/lib/armv7l/libinference_engine${CMAKE_SHARED_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/armv7l/libinference_engine_legacy${CMAKE_SHARED_LIBRARY_SUFFIX})
        endif()
    else()
        if(WITH_STATIC_LIB)
            set(DEPS ${OPENVINO_DIR}/lib/intel64/libinference_engine${CMAKE_STATIC_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/libinference_engine_legacy${CMAKE_STATIC_LIBRARY_SUFFIX})
        else()
            set(DEPS ${OPENVINO_DIR}/lib/intel64/libinference_engine${CMAKE_SHARED_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/libinference_engine_legacy${CMAKE_SHARED_LIBRARY_SUFFIX})
        endif()
    endif()
endif(WIN32)

if (NOT WIN32)
    set(DEPS ${DEPS}
        gflags yaml-cpp
        )
else()
    set(DEPS ${DEPS}
        gflags_static  libyaml-cppmt)
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
