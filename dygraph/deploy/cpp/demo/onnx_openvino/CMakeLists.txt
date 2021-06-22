# Openvino
include_directories("${OPENVINO_DIR}")
include_directories("${OPENVINO_DIR}/include")
include_directories("${OPENVINO_DIR}/external/tbb/include/tbb")
link_directories("${OPENVINO_DIR}/lib")
link_directories("${OPENVINO_DIR}/external/tbb/lib")

link_directories("${NGRAPH_LIB}/lib")

include_directories("${GFLAGS_DIR}/include")
link_directories("${GFLAGS_DIR}/lib")

if(WIN32)
    link_directories("${OPENVINO_DIR}/lib/intel64/Release")
    link_directories("${OPENVINO_DIR}/bin/intel64/Release")

    set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/Release/inference_engine${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/Release/inference_engine_legacy${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} gflags_static libyaml-cppmt shlwapi)
else()
    if (${ARCH} STREQUAL "armv7")
        if(WITH_STATIC_LIB)
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/armv7l/libinference_engine${CMAKE_STATIC_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/armv7l/libinference_engine_legacy${CMAKE_STATIC_LIBRARY_SUFFIX})
        else()
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/armv7l/libinference_engine${CMAKE_SHARED_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/armv7l/libinference_engine_legacy${CMAKE_SHARED_LIBRARY_SUFFIX})
        endif()
    else()
        if(WITH_STATIC_LIB)
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/libinference_engine${CMAKE_STATIC_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/libinference_engine_legacy${CMAKE_STATIC_LIBRARY_SUFFIX})
        else()
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/libinference_engine${CMAKE_SHARED_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${OPENVINO_DIR}/lib/intel64/libinference_engine_legacy${CMAKE_SHARED_LIBRARY_SUFFIX})
        endif()
    endif()
    set(DEPS ${DEPS} gflags yaml-cpp)
endif(WIN32)

message("-----DEPS = ${DEPS}")

# engine src
set(ENGINE_SRC ${PROJECT_SOURCE_DIR}/model_deploy/engine/src/openvino_engine.cpp)

add_executable(model_infer model_infer.cpp ${SRC} ${ENGINE_SRC} ${DETECTOR_SRC})
ADD_DEPENDENCIES(model_infer ext-yaml-cpp)
target_link_libraries(model_infer ${DEPS})
