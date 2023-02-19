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
macro (add_idealii_sources)
    file (RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND IDEAL_II_SRCS "${_relPath}/${_src}")
        else()
            list (APPEND IDEAL_II_SRCS "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SRCS to parent directory
        set (IDEAL_II_SRCS ${IDEAL_II_SRCS} PARENT_SCOPE)
    endif()
endmacro()