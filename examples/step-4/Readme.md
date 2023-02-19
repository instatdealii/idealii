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

# Example 4: Solving Navier-Stokes equations 

# Important note!
-----------------
This whole example is still work in progress, including this readme.
Currently the equations can only be solved with dG(0) as the temporal
derivative in the weak formulation is zero on each temporal element.

## Mathematical background
--------------------------

### Navier-Stokes equations
This example shows how to solve the Navier-Stokes equations, which is a coupled problem
between a vector valued velocity $\bf v\in\mathbb{R}^d$ and scalar valued pressure $p\in\mathbb{R}$ on a space-time cylinder with a d-dimensional spatial domain $\Omega\subset\mathbb{R}^d$. 

$$
\displaylines{
\partial_t \bf{v} - \nu\Delta\bf{v} +\bf{v}\cdot\nabla\bf{v} +\nabla p = f\text{ in }Q \\
\nabla\cdot \bf{v} = 0 \\
\bf{v} = g_D \text{ on }\Gamma_D\times I\\
\partial_n\bf{v} = 0\text{ on }\Gamma_N\times I\\
\bf{v} = \bf{v}^0 \text{ on }\Omega\times\{0\}
}
$$

Here, $\nu$ again denotes the kinematic viscosity. 

With the fully continuous function space $X=\{v\in L^2(I,H^1_D(\Omega)); \partial_t v\in L^2(I,H^1_D(\Omega)^{\*}); p\in L^2(I,L^2(\Omega)\}$
the fully continuous weak form reads as:   

Find $u=(\bf v,p)^T\in X+g_D$ such that:

### Problem description
Now we actually solve the Schäfer-Turek 2D-3 Benchmark including the nonlinear
convection term.

### Newton Linearization


## Technical details on ideal.II
------

### Evaluation of FE functions given by a (solution) vector
This functionality is still work in progress. 
When implemented it will be described here.