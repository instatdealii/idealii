## ---------------------------------------------------------------------
##
## Copyright (C) 2022 - 2023 by the ideal.II authors
##
## This file is part of the ideal.II library.
##
## The ideal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 3.0 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of ideal.II.
##
## ---------------------------------------------------------------------
set(TARGET "step-3")
set(TARGET_SRC step-3.cc)

cmake_minimum_required(VERSION 3.8)
if(CMAKE_BUILD_TYPE STREQUAL "" OR NOT CMAKE_BUILD_TYPE)
 set(CMAKE_BUILD_TYPE "Debug")
endif()

project(${TARGET} CXX)
include(CMakePrintHelpers)
find_package(ideal.II 0.8.0 
  HINTS ${ideal.II_DIR} ${IDEAL_II_DIR} ../ ../../ $ENV{IDEAL_II_DIR}
)
 
if(NOT ${ideal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of ideal.II. ***\n\n"
    "You may want to either pass a flag -DIDEAL_II_DIR=/path/to/ideal.II to cmake\n"
    "or set an environment variable \"IDEAL_II_DIR\" that contains this path."
  )
endif()
add_executable(${TARGET} ${TARGET_SRC})

target_link_libraries(${TARGET} ideal.II::ideal.II)