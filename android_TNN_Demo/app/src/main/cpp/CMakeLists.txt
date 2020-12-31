cmake_minimum_required(VERSION 3.10.2)
project("tengtnn")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
if (DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 20)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif ()

aux_source_directory(. SRC_LIST)

add_library( # Sets the name of the library.
        tengtnn

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        ${SRC_LIST})


find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# opencv
include_directories(
        ${CMAKE_SOURCE_DIR}/opencv/include/
)
add_library(libopencv_java4 STATIC IMPORTED)
set_target_properties(
        libopencv_java4
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/opencv/${ANDROID_ABI}/libopencv_java4.so
)

# tnn
include_directories(${CMAKE_SOURCE_DIR}/tnn/)
add_library(libTNN STATIC IMPORTED)
set_target_properties(
        libTNN
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/tnn/${ANDROID_ABI}/libTNN.so
)


target_link_libraries( # Specifies the target library.
        tengtnn

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        ${android-lib}
        ${jnigraphics-lib}
        libopencv_java4
        libTNN
        )

