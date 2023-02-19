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
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ideal.II/base/version.hh.in
  ${CMAKE_CURRENT_BINARY_DIR}/ideal.II/base/version.hh
)


## Install headers ############################################################
install(DIRECTORY ideal.II
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT library
  FILES_MATCHING
  PATTERN "*.hh"
)
 
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ideal.II
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT library
  FILES_MATCHING
  PATTERN *.hh
)

