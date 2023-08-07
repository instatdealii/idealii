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
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
cmake_policy(SET CMP0022 NEW)
project(ideal.II VERSION 0.9.0 LANGUAGES CXX)
option(DOCUMENTATION_ONLY "Only prepare Doxyfile and command" OFF)


include(CMakePrintHelpers)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(BUILD_SHARED_LIBS "Build deallib as a shared library" ON)

if(NOT DOCUMENTATION_ONLY)
  set(CMAKE_CXX_STANDARD 17)
  if(NOT CMAKE_BUILD_TYPE)
    message(WARNING "build type not set, using DebugRelease")
    set(CMAKE_BUILD_TYPE "DebugRelease")
  endif()
  # deal.II
  find_package(deal.II 9.3 QUIET
    HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
    )
  if(NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
      "*** Could not locate deal.II. ***\n\n"
      "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
      "or set an environment variable \"DEAL_II_DIR\" that contains this path."
      )
  endif()

  DEAL_II_INITIALIZE_CACHED_VARIABLES()
  cmake_print_variables(DEAL_II_TARGET_CONFIG)
  include(${DEAL_II_TARGET_CONFIG})

  option(IDEAL_II_COMPONENT_EXAMPLES
    "Enable configuration and installation of the example steps. This adds a COMPONENT \"examples\" to the build system."
    ON
  )


  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    list(APPEND IDEAL_II_BUILD_TYPES "debug")
    list(APPEND IDEAL_II_TARGETS ${PROJECT_NAME}_debug)
  endif()

  if(CMAKE_BUILD_TYPE MATCHES "Release")
    list(APPEND IDEAL_II_BUILD_TYPES "release")
    list(APPEND IDEAL_II_TARGETS ${PROJECT_NAME}_release)
  endif()

  if(NOT IDEAL_II_BUILD_TYPES)
      message(FATAL_ERROR "\n"
          "*** unsupported build type ***\n\n"
          "Please choose either Debug, Release or DebugRelease."
      )
  endif()

  add_subdirectory(cmake)
  add_subdirectory(include)
  add_subdirectory(src)

  foreach(build_type IN LISTS IDEAL_II_BUILD_TYPES)
          add_library(${PROJECT_NAME}_${build_type} ${IDEAL_II_SRCS})
          string(TOUPPER ${build_type} caps_build_type) 
          DEAL_II_SETUP_TARGET(${PROJECT_NAME}_${build_type} ${caps_build_type})
          target_include_directories(${PROJECT_NAME}_${build_type}
                  PUBLIC
                  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                  ${DEAL_II_INCLUDE_DIRS}
                  PRIVATE
                  ${CMAKE_SOURCE_DIR}/include
          )
          set_property(TARGET ${PROJECT_NAME}_${build_type} PROPERTY POSITION_INDEPENDENT_CODE ON)
          set_target_properties(${PROJECT_NAME}_${build_type}
                  PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
  endforeach()

    
  install(TARGETS ${IDEAL_II_TARGETS}
    EXPORT ${PROJECT_NAME}_Exports
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      NAMELINK_SKIP
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  install(EXPORT ${PROJECT_NAME}_Exports
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
    NAMESPACE ${PROJECT_NAME}::
  )

  install(TARGETS ${IDEAL_II_TARGETS}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      NAMELINK_ONLY
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
  )

  write_basic_package_version_file(
      "${PROJECT_NAME}ConfigVersion.cmake"
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY SameMajorVersion
  )

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
  )



  add_subdirectory(examples)
endif()
add_subdirectory(doc/doxygen)

