cmake_minimum_required(VERSION 3.0)
project(PaddleX CXX C)


option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   OFF)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
SET(OPENVINO_LIB "" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "" CACHE PATH "Location of libraries")

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

if (NOT DEFINED OPENVINO_LIB OR ${OPENVINO_LIB} STREQUAL "")
    message(FATAL_ERROR "please set OPENVINO_LIB with -DOPENVINO_LIB=/path/influence_engine")
endif()

if (NOT DEFINED OPENCV_DIR OR ${OPENCV_DIR} STREQUAL "")
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

include_directories("${OPENVINO_LIB}/")
include_directories("${OPENVINO_LIB}/include/")
link_directories("${OPENVINO_LIB}/lib")
link_directories("${OPENVINO_LIB}/externel/tbb/lib/")

if (WIN32)
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build/ NO_DEFAULT_PATH)
  unset(OpenCV_DIR CACHE)
else ()
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
endif ()

include_directories(${OpenCV_INCLUDE_DIRS})

if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
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

if (NOT WIN32)
    set(NGRAPH_PATH "${OPENVINO_LIB}")
    if(EXISTS ${NGRAPH_PATH})
        include(GNUInstallDirs)
        include_directories("${NGRAPH_PATH}/include")
        link_directories("${NGRAPH_PATH}/${CMAKE_INSTALL_LIBDIR}")
        set(NGRAPH_LIB ${NGRAPH_PATH}/${CMAKE_INSTALL_LIBDIR}/libngraph${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
endif()

if(WITH_STATIC_LIB)
	set(DEPS ${OPENVINO_LIB}/lib/intel64/libinference_engine${CMAKE_STATIC_LIBRARY_SUFFIX})
	set(DEPS ${DEPS} ${OPENVINO_LIB}/lib/intel64/libinference_engine_legacy${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
	set(DEPS ${OPENVINO_LIB}/lib/intel64/libinference_engine${CMAKE_SHARED_LIBRARY_SUFFIX})
	set(DEPS ${DEPS} ${OPENVINO_LIB}/lib/intel64/libinference_engine_legacy${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

if (NOT WIN32)
    set(DEPS ${DEPS}
        glog gflags  z  yaml-cpp
        )
else()
    set(DEPS ${DEPS}
        glog gflags_static libprotobuf zlibstatic xxhash libyaml-cppmt)
    set(DEPS ${DEPS} libcmt shlwapi)
endif(NOT WIN32)


if (NOT WIN32)
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

set(DEPS ${DEPS} ${OpenCV_LIBS})
add_executable(classifier src/classifier.cpp src/transforms.cpp src/paddlex.cpp)
ADD_DEPENDENCIES(classifier ext-yaml-cpp)
target_link_libraries(classifier ${DEPS})

