## ---------------------------------------------------------------------
##
## Copyright (C) 2024 by the ideal.II authors
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
option(IDEAL_II_SPHINX_DOCUMENTATION "Build sphinx documentation" ON)

if(IDEAL_II_SPHINX_DOCUMENTATION)
	find_package(Doxygen)
    find_package(Python3 COMPONENTS Interpreter)
	if( DOXYGEN_FOUND AND Python3_FOUND)
		set(DOXYGEN_IN Doxyfile.in)
		set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
		
		configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
		message(STATUS "Configured to build sphinx documentation")
		
		add_custom_target(sphinx 
            COMMAND ${CMAKE_COMMAND} -E make_directory _build
            COMMAND ${CMAKE_COMMAND} -E make_directory doxygenhtml
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/examples
            ${CMAKE_BINARY_DIR}/examples
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/_ext
            ${CMAKE_CURRENT_BINARY_DIR}/_ext
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/_static
            ${CMAKE_CURRENT_BINARY_DIR}/_static
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/examples
            ${CMAKE_CURRENT_BINARY_DIR}/examples
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/maths
            ${CMAKE_CURRENT_BINARY_DIR}/maths
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/conf.py 
            ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
            ${CMAKE_CURRENT_SOURCE_DIR}/installation.rst
            ${CMAKE_CURRENT_SOURCE_DIR}/make.bat
            ${CMAKE_CURRENT_SOURCE_DIR}/Makefile
            ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
            ${CMAKE_CURRENT_BINARY_DIR}
			COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT} > warnings_dox.out 2>&1
            COMMAND pip3 install -r requirements.txt 
            COMMAND make html > warnings_sphinx.out 2>&1
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			COMMENT "Generating Sphinx documentation"
			VERBATIM )
		
	else()
		message(STATUS "Doxygen or Python3 not found - skipping sphinx documentation")
	endif()


endif()
