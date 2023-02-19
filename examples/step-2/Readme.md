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

# Example 2: Solving Stokes equations 

## Mathematical background
-----------

### Stokes equation
This second example shows how to solve the Stokes equation, which is a coupled problem
between a vector valued velocity $\bf v\in\mathbb{R}^d$ and scalar valued pressure $p\in\mathbb{R}$ on a space-time cylinder with a d-dimensional spatial domain $\Omega\subset\mathbb{R}^d$. 

$$
\displaylines{
\partial_t \bf{v} - \nu\Delta\bf{v} +\nabla p = f\text{ in }Q \\
\nabla\cdot \bf{v} = 0 \\
\bf{v} = g_D \text{ on }\Gamma_D\times I\\
\partial_n\bf{v} = 0\text{ on }\Gamma_N\times I\\
\bf{v} = \bf{v}^0 \text{ on }\Omega\times\{0\}
}
$$

Here, $\nu$ denotes the kinematic viscosity. 

With the fully continuous function space $X=\{v\in L^2(I,H^1_D(\Omega)); \partial_t v\in L^2(I,H^1_D(\Omega)^{\*}); p\in L^2(I,L^2(\Omega))\}$
the fully continuous weak form reads as:   

Find $u=(\bf v,p)^T\in X+g_D$ such that:

$$
\displaylines{
 (\partial_t \bf{v},\varphi^{\bf{v}}) + \nu(\nabla_x {\bf{v}},\nabla_x\varphi^{\bf{v}}) - (p,\nabla_x\cdot \varphi^{\bf{v}}) + (\nabla_x\cdot\bf{v},\varphi^p) = 0 \\
 \bf{v}(0,x) = \bf{v}^0(x) \text{ in }\Omega
 }
$$

Where $H^1_D(\Omega)$ is the space of functions with zero Dirichlet contraints on the Dirichlet boundary $\Gamma_D$.
Note that for problems with pure Dirichlet constraints, i.e. $\Gamma_D\equiv\partial\Omega$, an additional compatibilty constraint for the pressure variable is needed.

### Discretization
Discretization follows almost the same steps as for the heat equation.
The main difference is the spatial discretization of the function space $X$ into the inf-sup stable Taylor-Hood pair $Q_2Q_1$
i.e. continuous biquadratic functions for the velocity $\bf{v}$ and continuous bilinear functions for the pressure $p$.

### Problem description
We solve a variation of the Schäfer-Turek 2D-3 Benchmark which usually solves the Navier-Stokes equations.
The only difference is that we (for now) omit the nonlinear convection term. 
Everything else is unchanged such that we still use the same domain/geometry and boundary conditions.
Which for the velocity are Poisseulle inflow conditions on the left boundary, homogeneous Neumann conditions (outflow) 
on the right boundary and no-slip (zero Dirichlet) constraints on the other two boundaries and the obstacle.

## Technical details on ideal.II
------

### Handling non-zero Dirichlet boundary conditions.
In principle this is not that different from the previous example. Instead of calling `interpolate_boundary_values` 
with the build in `dealii::ZeroFunction` we now supply our own function `PoisseuileInflow` which describes the quadratic
inflow profile on the left boundary (id 0) of the domain and is scaled in time by a sine function. 

The important difference to handling non-zero boundary conditions in stationary problems is the use of the functions 
`set_time` and `get_time` to account for time-dependency. These are already available from deal.II and are used internally in the ideal.II version `interpolate_boundary_values` and similar functions.
For that reason you should use `get_time` to obtain the temporal variable. 


### Handling vector valued problems
This is very similar to the stationary case. Instead of just using `FESystem` this is handed over to the spacetime
finite element class as the spatial element description. 

The most important difference is in how deal.II and ideal.II use Extractors for evaluation of shape functions.
In deal.II this is done by using the supplied `operator[]` on the FEValues object i.e

```
const FEValuesExtractors::Vector velocities(0);
dealii::Tensor<1,dim> phi_v = fe_values[velocities].value(i,q);
```

`fe_values[velocities]` returns a reference to a `dealii::FEValuesViews::Vector` object which is stored in a local
cache of the `fe_values` object. 

To reproduce this behaviour ideal.II would need to store a cache of caches which seemed unnecessarily complicated and 
costly to store and compute. 
Instead ideal.II internally uses the function call from the code block above and directly supplies appropriate functions. 

Then, the call to obtain the shape value of the space-time velocity basis functions is

```
const FEValuesExtractors::Vector velocities(0);
fe_values_spacetime.vector_value(velocity,i,q)
```

With $\varphi_t$ as the temporal and $\varphi_x^{\bf{v}}$ and $\varphi_x^p$ as the spatial basis functions, belonging to the velocity and pressure extractor respectively, the function evaluations compute the following shape functions

- `vector_value(velocity,i,q)` calculates $\varphi_t\varphi_x^{\bf{v}}$
- `vector_dt(velocity,i,q)` calculates $\partial_t\varphi_t\varphi_x^{\bf{v}}$
- `vector_space_grad(velocity,i,q)` calculates $\varphi_t\nabla_x\varphi_x^{\bf{v}}$
- `vector_divergence(velocity,i,q)` calculates $\varphi_t\nabla_x\cdot\varphi_x^{\bf{v}}$
- `scalar_value(pressure,i,q)` calculates $\varphi_t\varphi_x^{p}$

