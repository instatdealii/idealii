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
if(IDEAL_II_COMPONENT_EXAMPLES)
  message(STATUS "Setting up examples")
  add_custom_target(examples)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION examples
    COMPONENT examples
    FILES_MATCHING
    #
    # EXCLUDES
    # PATTERN <pattern> EXCLUDE
    #
    # Glob includes:
    #
    PATTERN "*cc"
    PATTERN "step*/CMakeLists.txt"
    PATTERN "*.inp"
    PATTERN "*.md"
  )
    
  set(IDEAL_II_COMPILE_EXAMPLES FALSE) #TODO: fix problem with linking deal.II lib itself!
  if(IDEAL_II_COMPILE_EXAMPLES) 
	  file(GLOB _steps
	  	${CMAKE_CURRENT_SOURCE_DIR}/step-*/step-*.cc)
	  
	  foreach(_step ${_steps})
	    get_filename_component(_name ${_step} NAME_WE)
	    get_filename_component(_directory ${_step} DIRECTORY)  
	  
	  	#
	  	# TODO: extract deal.II dependency info
	  	#
	  	set(_setup FALSE)
	  	set(_setup TRUE)
	  	
	  	if(_setup)
	  	  message(STATUS " ${_name} - starting setup" )
	  	  foreach(_build ${IDEAL_II_BUILD_TYPES})
	  	  
		  string(TOUPPER ${_build} _build_upper) 
	  	  set(_target "example_${_name}_${_build}")
	  	  #insource setup target!
	  	  add_executable(${_target} ${_step})
	  	  
	      target_link_libraries(${_target} ${DEAL_II_TARGET_${_build_upper}})
	      target_link_libraries(${_target} ideal.II_${_build})
	   
		  get_property(DEAL_II_LINK_LIBS TARGET ${DEAL_II_TARGET_${_build_upper}} PROPERTY INTERFACE_LINK_LIBRARIES)
	
			target_link_libraries(${_target} ${DEAL_II_LINK_LIBS})
	 	   set_target_properties(${_target}
		     PROPERTIES 
	         RUNTIME_OUTPUT_NAME "${_name}.${_build}"
			 RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
	 	   )
			
	  	  get_property(DEAL_II_LIB_LOCATION TARGET ${DEAL_II_TARGET_${_build_upper}} PROPERTY LOCATION)
	      cmake_print_variables(DEAL_II_LIB_LOCATION)
	      set_target_properties(${_target}
	      	PROPERTIES 
	      	INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
		  )
		  add_dependencies(examples ${_target})
		  install(TARGETS ${_target}
		  	DESTINATION examples/${_name}
	  	  )
	  	  endforeach()
	  	else()
	  		message(STATUS " ${_name} - dependencies not satisfied")    	
	  	endif()
	  	
	  	
	  endforeach()
  endif()
  
endif()