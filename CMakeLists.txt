cmake_minimum_required(VERSION 2.8.12)
project(felsenstein C CXX)
set(HAVE_AVX2 0 CACHE BOOL "Have AVX2")
set(HAVE_SSE4_1 0 CACHE BOOL "Have SSE4.1")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindASan)
include(FindMSan)
set(CMAKE_C_STANDARD 99)

IF(DEBUG_PRINT)
    ADD_DEFINITIONS(-DDEBUG_PRINT)
ENDIF(DEBUG_PRINT)

if ("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
    message("-- Compiler is clang(++)")
    set(CMAKE_COMPILER_IS_CLANG 1)
endif()

if (HAVE_AVX2)
    if (CMAKE_COMPILER_IS_CLANG)
        set(SSE_FLAGS "-DAVX2=1 -mavx2")
    else ()
        set(SSE_FLAGS "-DAVX2=1 -mavx2 -Wa,-q")
    endif ()
elseif (HAVE_SSE4_1)
    set(SSE_FLAGS "-DSSE=1 -msse4.1")
else ()
    include(CheckSSEFeatures)
    if (HAVE_AVX2_EXTENSIONS)
        set(SSE_FLAGS "${SSE_FLAGS} -DAVX2=1")
    else ()
        if (HAVE_SSE4_1_EXTENSIONS)
            set(SSE_FLAGS "${SSE_FLAGS} -DSSE=1")
        else ()
            message(FATAL_ERROR "At least SSE4.2 is needed to compile!")
        endif ()
    endif ()
endif ()

include_directories(lib)

add_executable(simd_test simd_test.c simd_functions_ps.h)
set_target_properties(simd_test PROPERTIES COMPILE_FLAGS ${SSE_FLAGS} LINK_FLAGS ${SSE_FLAGS})


add_library(felsenstein felsenstein_lin.c felsenstein_lin.h)
add_library(felsenstein_log felsenstein_logspace.c felsenstein_lin.h)

add_executable(felsenstein_test felsenstein_lin.h felsenstein_lin.c felsenstein_test.c)
target_link_libraries(felsenstein_test m)

add_executable(felsenstein_log_test felsenstein_lin.h felsenstein_logspace.c felsenstein_logtest.c)
target_link_libraries(felsenstein_log_test m)

add_executable(felsenstein_log_test_float felsenstein_logspace_float.c)
target_link_libraries(felsenstein_log_test m)

add_executable(felsenstein_log_test_float_log2 felsenstein_logspace_float_log2.c)
target_link_libraries(felsenstein_log_test_float_log2 m)

add_executable(felsenstein_faster_test_float felsenstein_faster.c felsenstein_logtest_faster.c)
target_compile_definitions(felsenstein_faster_test_float PUBLIC SINGLE_PRECISION)
target_link_libraries(felsenstein_faster_test_float m)

add_executable(felsenstein_faster_test felsenstein_faster.c felsenstein_logtest_faster.c)
target_compile_definitions(felsenstein_faster_test PUBLIC DOUBLE_PRECISION)
target_link_libraries(felsenstein_faster_test m)

add_executable(felsenstein_log_test16 felsenstein_lin.h felsenstein_logspace.c felsenstein_logtest16.c)
target_link_libraries(felsenstein_log_test16 m)

add_executable(felsenstein_test16 felsenstein_lin.h felsenstein_lin.c felsenstein_test16.c)
target_link_libraries(felsenstein_test16 m)

add_executable(felsenstein_simd_test_float felsenstein_simd.c felsenstein_logtest_simd.c)
set_target_properties(felsenstein_simd_test_float PROPERTIES COMPILE_FLAGS ${SSE_FLAGS} LINK_FLAGS ${SSE_FLAGS})
target_compile_definitions(felsenstein_simd_test_float PUBLIC SINGLE_PRECISION)
target_link_libraries(felsenstein_simd_test_float m)

add_executable(felsenstein_simd_test felsenstein_simd.c felsenstein_logtest_simd.c)
set_target_properties(felsenstein_simd_test PROPERTIES COMPILE_FLAGS ${SSE_FLAGS} LINK_FLAGS ${SSE_FLAGS})
target_compile_definitions(felsenstein_simd_test PUBLIC DOUBLE_PRECISION)
target_link_libraries(felsenstein_simd_test m)

add_executable(test_sind_functions test_simd_functions_pd.c)
set_target_properties(test_sind_functions PROPERTIES COMPILE_FLAGS ${SSE_FLAGS} LINK_FLAGS ${SSE_FLAGS})
target_link_libraries(test_sind_functions m)