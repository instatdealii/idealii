<!---
---------------------------------------------------------------------
Copyright (C) 2022 - 2023 by the ideal.II authors

This file is part of the ideal.II library.

The ideal.II library is free software; you can use it, redistribute
it, and/or modify it under the terms of the GNU Lesser General
Public License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.
The full text of the license can be found in the file LICENSE.md at
the top level directory of ideal.II.
---------------------------------------------------------------------
--->

# Overview over the example steps
--------------------------------- 

## step-1 Heat equation 
-----------------------
This step serves as an introduction to tensor-product space-time finite elements
and the basic structure of the library. 
It is recommended to at least know the deal.II tutorial examples [1](https://dealii.org/current/doxygen/deal.II/step_1.html) to [3](https://dealii.org/current/doxygen/deal.II/step_3.html).

## step-2 Stokes equations
-------------------------
This step introduces the handling of coupled equations with vector-valued components in ideal.II.
Additionally, handling of nonhomogeneous Dirichlet boundary conditions is introduced.
It is recommended to know the deal.II tutorial for solving the stationary Stokes equations, i.e. [22](https://dealii.org/current/doxygen/deal.II/step_22.html)
or any other example program handling coupled equations.


## step-3 Heat equation with MPI parallel linear algebra provided by Trilinos
-----------------------------------------------------------------------------
This step shows how to parallelize your code and especially how to handle locally owned and 
ghost degrees of freedom. 
Additionally, nonzero initial conditions and Legendre support points in time are introduced.
For this example familiarity with parallel deal.II examples is recommended, e.g. step [40](https://dealii.org/current/doxygen/deal.II/step_40.html)
 
(Note that support for PetSc is not planned as global space-time indices would have to be renumbered to 
account for the constraint of contiguous index sets in PetSc.)

## step-4 (in progress) Navier-Stokes equations with MPI
-----------------------------------------
This step will show how to handle coupled equations with MPI and how to handle nonlinear equations
with a space-time Newton solver.

This step is working for dG(0) 

## step-5 (planned) Heat equation with space-time adaptivity using partition-of-unity localization on dual weighted residuals
-----------------------------------------------------------------------------------------------------------------------------
This step will show 