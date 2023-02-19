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
option(IDEAL_II_DOCUMENTATION "Build doxygen documentation" ON)

if(IDEAL_II_DOCUMENTATION)
	find_package(Doxygen)
	if( DOXYGEN_FOUND )
		set(DOXYGEN_IN Doxyfile.in)
		set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
		
		configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
		message(STATUS "Configured to build doxygen documentation")
		
		add_custom_target(doxygen 
			COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT} > warnings.out 2>&1
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINDARY_DIR}
			COMMENT "Generating Doxygen docymentation"
			VERBATIM )
	else()
		message(STATUS "Doxygen not found - skipping documentation")
	endif()


endif()