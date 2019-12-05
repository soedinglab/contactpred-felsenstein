cmake_minimum_required(VERSION 3.14)
project(felsenstein)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindASan)
set(CMAKE_C_STANDARD 99)

add_library(felsenstein felsenstein.c felsenstein.h)
add_executable(felsenstein_test felsenstein.h felsenstein.c dummy_test.c)

add_executable(felsenstein_log_test felsenstein.h felsenstein_logspace.c dummy_test.c)