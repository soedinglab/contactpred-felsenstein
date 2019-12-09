cmake_minimum_required(VERSION 3.14)
project(felsenstein)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindASan)
set(CMAKE_C_STANDARD 99)

add_library(felsenstein felsenstein.c felsenstein.h)
add_library(felsenstein_log felsenstein_logspace.c felsenstein.h)

add_executable(felsenstein_test felsenstein.h felsenstein.c felsenstein_test.c)
target_link_libraries(felsenstein_test m)

add_executable(felsenstein_log_test felsenstein.h felsenstein_logspace.c felsenstein_logtest.c)
target_link_libraries(felsenstein_log_test m)

add_executable(felsenstein_log_test16 felsenstein.h felsenstein_logspace.c felsenstein_logtest16.c)
target_link_libraries(felsenstein_log_test16 m)

add_executable(felsenstein_test16 felsenstein.h felsenstein.c felsenstein_test16.c)
target_link_libraries(felsenstein_test16 m)