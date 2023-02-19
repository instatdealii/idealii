// ---------------------------------------------------------------------
//
// Copyright (C) 2022 - 2023 by the ideal.II authors
//
// This file is part of the ideal.II library.
//
// The ideal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of ideal.II.
//
// ---------------------------------------------------------------------

#include <ideal.II/base/version.hh>
#include <ideal.II/base/idealii.hh>

#include <fstream>
#include <iostream>

namespace idealii{
  
  void print_version_info(){
    std::cout << "This is ideal.II in version v"
      << IDEAL_II_VERSION_MAJOR << "."
      << IDEAL_II_VERSION_MINOR << "."
      << IDEAL_II_VERSION_PATCH;
#ifdef DEBUG
      std::cout << " in DEBUG mode";
#endif
    std::cout << std::endl;
  }
}//namespace deallib
