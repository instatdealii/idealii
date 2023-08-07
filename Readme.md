<!---
Copyright (C) 2022 - 2023 by the ideal.II authors

This file is part of the ideal.II library.

The ideal.II library is free software; you can use it, redistribute
it, and/or modify it under the terms of the GNU Lesser General
Public License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.
The full text of the license can be found in the file LICENSE.md at
the top level directory of ideal.II.
--->

# ideal.II an extension to deal.II for tensor-product space-time finite elements
This library provides classes and functions to solve instationary partial differential equations by 
using space-time finite elements i.e. finite elements in both time and space. 
It heavily builds on the 1 to 3d finite element capabilities of [deal.II](www.dealii.org) 
so you should make yourself familiar with solving stationary problems with that library first.


## How to install ideal.II
First you need to install deal.II by following the instructions on their [site](https://dealii.org/current/readme.html).
If you plan on using parallel linear algebra and other advanced features of deal.II and ideal.II 
you should use either the [cmake superbuild](www.github.com/jpthiele/dealii-cmake-superbuild) or [candi](www.github.com/dealii/candi). 
Note down the installation directory of deal.II (for example ~/Software/dealii)

The installation of ideal.II is based on CMake as well, the steps are as follows.

1. Choose an installation directory for ideal.II (for example ~/Software/idealii)

2. Open a console and go to a directory where you want to download the source files (e.g. ~/Downloads) and clone the repository 

~~~~~
    cd ~/Downloads
    git clone https://github.com/instatdealii/idealii
~~~~~

3. After cloning is finished execute the following commands (with the above directory examples write
  ~/Software/dealii instead of <path_to_your_deal_installation> and ~/Software/idealii instead of <path_to_install_idealii_in>
 
~~~~~
    cd idealii   
    cmake -S. -Bbuild -DDEAL_II_DIR=<path_to_your_deal_installation> -DCMAKE_INSTALL_PREFIX=<path_to_install_idealii_in> 
    cmake --build build
    cmake --install build
~~~~~

## How to use ideal.II in your codes
Take a look at one of the example steps and copy the CMakeLists.txt file from there into your project directory. 
Set the correct project name and sources and then call the following commands from your project directory.

~~~~~
    cmake -S. -Bbuild -DIDEAL_II_DIR=<path_to_install_idealii_in>
    cmake --build build
~~~~~

Then the executable will be inside the build subdirectory. 

## How to cite ideal.II
The software metapaper is currently in preparation. Once published the readme will be updated accordingly.

If you write a paper using results obtained with the help of ideal.II, please cite the following reference: 

1. Jan Philipp Thiele, 
   ideal.II: a Galerkin space-time extension to the finite element library deal.II,
   2023, in preparation
   
~~~~~
      @article{idealII,
               title = {\texttt{ideal.II}: a Galerkin space-time extension to the finite element library deal.II},
               author = {Jan Philipp Thiele},
               year = {2023,\textit{in preparation}}
      }
~~~~~
   
2. Please also cite deal.II as explained [here](https://dealii.org/publications.html)
    
## Doxygen documentation
The doxygen documentation of the library functions can be found 
[here](https://instatdealii.github.io/idealii).



 
