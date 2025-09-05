#    This file will define the following variables for find_package method:
#      - UltraInfer_LIBS                 : The list of libraries to link against.
#      - UltraInfer_INCLUDE_DIRS         : The UltraInfer include directories.
#      - UltraInfer_Found                : The status of UltraInfer

include(${CMAKE_CURRENT_LIST_DIR}/UltraInfer.cmake)
# setup UltraInfer cmake variables
set(UltraInfer_LIBS ${ULTRAINFER_LIBS})
set(UltraInfer_INCLUDE_DIRS ${ULTRAINFER_INCS})
set(UltraInfer_FOUND TRUE)    
