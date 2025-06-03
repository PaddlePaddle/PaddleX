# This function comes from https://blog.csdn.net/yindongjie1221/article/details/90614261
function(redefine_file_macro targetname)
    get_target_property(source_files "${targetname}" SOURCES)
    foreach(sourcefile ${source_files})
        get_property(defs SOURCE "${sourcefile}"
            PROPERTY COMPILE_DEFINITIONS)
        get_filename_component(filepath "${sourcefile}" ABSOLUTE)
        string(REPLACE ${PROJECT_SOURCE_DIR}/ "" relpath ${filepath})
        list(APPEND defs "__REL_FILE__=\"${relpath}\"")
        set_property(
            SOURCE "${sourcefile}"
            PROPERTY COMPILE_DEFINITIONS ${defs}
            )
    endforeach()
endfunction()

function(download_and_decompress url filename decompress_dir)
  if(NOT EXISTS ${filename})
    message("Downloading file from ${url} to ${filename} ...")
    file(DOWNLOAD ${url} "${filename}.tmp" SHOW_PROGRESS)
    file(RENAME "${filename}.tmp" ${filename})
  endif()
  if(NOT EXISTS ${decompress_dir})
    file(MAKE_DIRECTORY ${decompress_dir})
  endif()
  message("Decompress file ${filename} ...")
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${filename} WORKING_DIRECTORY ${decompress_dir})
endfunction()

function(get_openvino_libs OPENVINO_RUNTIME_DIR)
  set(LIB_LIST "")
  find_library(OPENVINO_LIB openvino PATHS ${OPENVINO_RUNTIME_DIR}/lib/ ${OPENVINO_RUNTIME_DIR}/lib/intel64 NO_DEFAULT_PATH)
  list(APPEND LIB_LIST ${OPENVINO_LIB})

  set(TBB_DIR ${OPENVINO_RUNTIME_DIR}/3rdparty/tbb/lib/cmake)
  message(STATUS "TBB_DIR: ${TBB_DIR}")
  find_package(TBB PATHS ${TBB_DIR})
  if (TBB_FOUND)
    # 2024.10.22(zhangyue): Use openvino with tbb on linux
    set(TBB_LIB "${OPENVINO_RUNTIME_DIR}/3rdparty/tbb/lib/libtbb.so")
    list(APPEND LIB_LIST ${TBB_LIB})
  else()
    # TODO(zhoushunjie): Use openvino with tbb on linux in future.
    set(OMP_LIB "${OPENVINO_RUNTIME_DIR}/3rdparty/omp/lib/libiomp5.so")
    list(APPEND LIB_LIST ${OMP_LIB})
  endif()
  set(OPENVINO_LIBS ${LIB_LIST} PARENT_SCOPE)
endfunction()

function(remove_duplicate_libraries libraries)
  list(LENGTH ${libraries} lib_length)
  set(libraries_temp "")
  set(full_libraries "")
  foreach(lib_path ${${libraries}})
    get_filename_component(lib_name ${lib_path} NAME)
    list(FIND libraries_temp ${lib_name} lib_idx)
    if (${lib_idx} EQUAL -1)
      list(APPEND libraries_temp ${lib_name})
      list(APPEND full_libraries ${lib_path})
    endif()
  endforeach()
  set(${libraries} ${full_libraries} PARENT_SCOPE)
endfunction()

function(get_windows_path win_path origin_path)
  STRING(REGEX REPLACE "/" "\\\\" _win_path ${origin_path})
  set(${win_path} ${_win_path} PARENT_SCOPE)
endfunction()

function(get_osx_architecture)
  if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(CURRENT_OSX_ARCH "arm64" PARENT_SCOPE)
  elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    set(CURRENT_OSX_ARCH "x86_64" PARENT_SCOPE)
  else()
    set(CURRENT_OSX_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR} PARENT_SCOPE)
  endif()
endfunction()


# A fake target to include all the libraries and tests the ultra_infer module depends.
add_custom_target(fd_compile_deps COMMAND echo 1)

# A function to grep LINK_ONLY dependencies from INTERFACE_LINK_LIBRARIES
function(regrex_link_only_libraries OUTPUT_DEPS PUBLIC_DEPS)
  string(JOIN "#" _public_deps ${PUBLIC_DEPS})
  string(REPLACE "$<LINK_ONLY:" "" _public_deps ${_public_deps})
  string(REPLACE ">" "" _public_deps ${_public_deps})
  string(REPLACE "#" ";" _public_deps ${_public_deps})
  set(${OUTPUT_DEPS} ${_public_deps} PARENT_SCOPE)
endfunction()

# Bundle several static libraries into one. This function is modified from Paddle Lite. 
# reference: https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/cmake/lite.cmake#L252
function(bundle_static_library tgt_name bundled_tgt_name fake_target)
  list(APPEND static_libs ${tgt_name})
  add_dependencies(fd_compile_deps ${fake_target})
  # Set redundant static libs here, protobuf is already available 
  # in the Paddle Lite static library. So, we don't need protobuf 
  # in opencv. And there is no need for opencv_dnn, opencv_ml, 
  # opencv_flann and some other modules. Therefore, we chose
  # to discard these redundant modules.
  set(REDUNDANT_STATIC_LIBS opencv_dnn opencv_calib3d opencv_photo 
      opencv_flann opencv_objdetect opencv_stitching opencv_gapi 
      opencv_ml libprotobuf)

  function(_recursively_collect_dependencies input_target)
    list(FIND REDUNDANT_STATIC_LIBS ${input_target} _input_redunant_id)
    if(${_input_redunant_id} GREATER 0)
      return()
    endif()
    set(_input_link_libraries LINK_LIBRARIES)
    # https://cmake.org/cmake/help/latest/prop_tgt/TYPE.html
    get_target_property(_input_type ${input_target} TYPE)
    # In OpenCVModules.cmake, they set the deps of modules
    # (opencv_core,...) as INTERFACE_LINK_LIBRARIES. The 
    # 'Type' of opencv static lib is set as 'STATIC_LIBRARY'.
    if ((${_input_type} STREQUAL "INTERFACE_LIBRARY")
         OR (${_input_type} STREQUAL "STATIC_LIBRARY"))
      set(_input_link_libraries INTERFACE_LINK_LIBRARIES)
    endif()
    get_target_property(_public_dependencies ${input_target} ${_input_link_libraries})
    regrex_link_only_libraries(public_dependencies "${_public_dependencies}")
    
    foreach(dependency IN LISTS public_dependencies)
      if(TARGET ${dependency})
        get_target_property(alias ${dependency} ALIASED_TARGET)
        if (TARGET ${alias})
          set(dependency ${alias})
        endif()
        get_target_property(_type ${dependency} TYPE)
        list(FIND REDUNDANT_STATIC_LIBS ${dependency} _deps_redunant_id)
        if (${_type} STREQUAL "STATIC_LIBRARY" AND 
            (NOT (${_deps_redunant_id} GREATER 0)))
          list(APPEND static_libs ${dependency})
        endif()

        get_property(library_already_added
          GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency})
        if (NOT library_already_added)
          set_property(GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency} ON)
          if(NOT (${_deps_redunant_id} GREATER 0))
            _recursively_collect_dependencies(${dependency})
          endif()
        endif()
      endif()
    endforeach()
    set(static_libs ${static_libs} PARENT_SCOPE)
  endfunction()

  _recursively_collect_dependencies(${tgt_name})

  list(REMOVE_DUPLICATES static_libs)
  list(REMOVE_ITEM static_libs ${REDUNDANT_STATIC_LIBS})
  message(STATUS "WITH_STATIC_LIB=${WITH_STATIC_LIB}, Found all needed static libs from dependency tree: ${static_libs}")
  message(STATUS "Exclude some redundant static libs: ${REDUNDANT_STATIC_LIBS}")

  set(bundled_tgt_full_name
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX})

  message(STATUS "Use bundled_tgt_full_name:  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")

  if(WIN32)
    message(FATAL_ERROR "Not support UltraInfer static lib for windows now.")
  endif()

  add_custom_target(${fake_target} ALL COMMAND ${CMAKE_COMMAND} -E echo "Building fake_target ${fake_target}")
  add_dependencies(${fake_target} ${tgt_name})
  # add_dependencies(${fake_target} fastdelpoy_dummy)

  if(NOT IOS AND NOT APPLE)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in
      "CREATE ${bundled_tgt_full_name}\n" )

    foreach(tgt IN LISTS static_libs)
      file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in
        "ADDLIB $<TARGET_FILE:${tgt}>\n")
    endforeach()

    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in "SAVE\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in "END\n")

    file(GENERATE
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar
      INPUT ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in)

    set(ar_tool ${CMAKE_AR})
    if (CMAKE_INTERPROCEDURAL_OPTIMIZATION)
      set(ar_tool ${CMAKE_CXX_COMPILER_AR})
    endif()
    message(STATUS "Found ar_tool: ${ar_tool}")

    add_custom_command(
      TARGET ${fake_target} PRE_BUILD
      COMMAND rm -f ${bundled_tgt_full_name}
      COMMAND ${ar_tool} -M < ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar
      COMMENT "Bundling ${bundled_tgt_name}"
      COMMAND ${CMAKE_STRIP} --strip-unneeded ${CMAKE_CURRENT_BINARY_DIR}/lib${bundled_tgt_name}.a
      COMMENT "Stripped unneeded debug symbols in ${bundled_tgt_name}"
      DEPENDS ${tgt_name}
      VERBATIM)
  else()
    foreach(lib ${static_libs})
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    add_custom_command(
      TARGET ${fake_target} PRE_BUILD
      COMMAND rm -f ${bundled_tgt_full_name}
      COMMAND /usr/bin/libtool -static -o ${bundled_tgt_full_name} ${libfiles}
      COMMENT "Bundling ${bundled_tgt_name}"
      COMMAND ${CMAKE_STRIP} -S ${CMAKE_CURRENT_BINARY_DIR}/lib${bundled_tgt_name}.a
      COMMENT "Stripped unneeded debug symbols in ${bundled_tgt_name}"
      DEPENDS ${tgt_name}
    )
  endif()

  add_library(${bundled_tgt_name} STATIC IMPORTED GLOBAL)
  set_property(TARGET ${bundled_tgt_name} PROPERTY IMPORTED_LOCATION
                                         ${bundled_tgt_full_name})          
  add_dependencies(${bundled_tgt_name} ${fake_target})
  add_dependencies(${bundled_tgt_name} ${tgt_name})

endfunction()
