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

# Example 1: Solving the heat equation 

## How to build
---------------
The recommended build type is "out of source" and will work as follows
1. Navigate into the step-1 folder in Terminal/Console
2. Execute the following CMake command (depending on your ideal.II install location)
  
```
  cmake -S. -Bbuild -DIDEAL_II_DIR=<path_to_your_idealii_install>
```
  
3. The executable will then be in the build directory and can be run by

```  
  cd build
  ./step-1
```

## Mathematical background
-----------

### Heat equation
This first example shows how to solve the heat equation in a space-time cylinder, i.e. the 
tensor product $Q = \Omega\times I$ with spatial domain $\Omega$ and temporal domain/interval $I=(0,T)$.

$$ 
\displaylines{
\partial_t u - \Delta u = f\text{ in }Q\\
u = g \text{ on }\partial\Omega\times I\\
u = u^0 \text{ on }\Omega\times\{0\}
}
$$

With the fully continuous function space $X=\{v\in L^2(I,H^1_0(\Omega));\partial_t v\in L^2(I,H^1_0(\Omega)^*) \}$
the fully continuous weak form reads as:   
Find $u\in X+g$ such that:

$$
\displaylines{
 \int\limits_0^T\int\limits_\Omega \partial_t u\varphi + \nabla_x u\nabla_x\varphi \mathrm{d}x\mathrm{d}t
 = \int\limits_0^T\int\limits_\Omega f\varphi  \mathrm{d}x\mathrm{d}t \forall\varphi\in X\\
 u(0,x) = u^0(x) \text{ in }\Omega
}
$$

### Discretization
Using a temporal (1d) triangulation into intervals $T_k =\cup_{m=1}^M I_m$, with $I_m\coloneqq [t_{m-1},t_m]$, and a spatial (2d) triangulation $T_h$ we discretize $Q$ as $Q_{kh}=t_h\times I_k$.

This is the simplest case, for which `fixed::Triangulation<dim>`  is written. 
The usage of multiple spatial meshes over time, so called dynamic meshes, will be 
discussed in later examples with adaptive refinement.

To allow for discontinuities in time we introduce:
- the limits from below $u^-(t_m)=\lim\limits_{t\nearrow t_m} u(t)$ and above $u^+(t_m)=\lim\limits_{t\searrow t_m} u(t)$, 
- as well as the jump term $[u]_m=u^+(t_m)-u^-(t_m)$
  
The resulting fully discrete function space is:

$$
\tilde{X}_{kh}^{rs}=\lbrace u\in L^2(I,L^2(\Omega)); u|_{I_m} \in \mathcal{P}_r(I_m)\otimes V_h^s(T_h)\rbrace
$$ 

and the discrete weak form reads as:   
Find $u_{kh}\in \tilde{X}_{kh}^{rs}+g$ such that:

$$
\displaylines{
\sum\limits_{m=1}^M \int\limits_{t_{m-1}}^{t_m} \int\limits_\Omega \partial_t u_{kh}\varphi_{kh}+\nabla_x u_{kh}\nabla_x\varphi_{kh} \mathrm{d}x\mathrm{d}t +\int\limits_\Omega [u_{kh}]_{m-1}  \varphi_{kh}^+(t_{m-1}) \mathrm{d}x
=\sum\limits_{m=1}^M \int\limits_{t_{m-1}}^{t_m} \int\limits_\Omega f\varphi_{kh}\mathrm{d}x\mathrm{d}t
\forall\varphi_{kh}\in \tilde{X}_{k,h}^{r,s}\\
u_{kh}(0,x) = u^0(x) \text{ in }\Omega
}
$$

### Discretization with a tensor-product slab
Solving the above weak form on the whole space-time cylinder we obtain a so called full space-time method.
Solving it in sequence an all subintervals of $T_k$ we obtain a form of time-stepping method, but without
explicit derivation of a Butcher tableau. 
To allow for using both methods or something in between we introduce a **slab**, which is the tensor product 
between a contiguous subset of $T_h$ and $\Omega$. 

Then, using the complete temporal mesh the slab corresponds to the full space-time method while using single interval slabs corresponds to time-stepping. As a consequence this definition allows for the maximum flexibility. 

## Technical details on ideal.II
------

### The `slab::` namespace
By the above definition the slab namespace has objects for use on a single slab.
At the moment these are a DoFHandler and Triangulation classes. 
In essence these objects are a combination of a spatial n-dimensional and a temporal 1-dimensional deal.II object
with some functions for easier usage.

Note that we don't need any special clases for vectors and matrices as we simply use the deal.II objects with
space-time indexing.
Additionally, the namespace contains DoFTools and VectorTools namespaces that offer similar functionality compared to 
the same deal.II namespaces.

### The `spacetime::` namespace
Objects in the spacetime namespace fall into different categories.

The first category are collections of slab classes as std::list, 
this collection type allows for easy insertion of new objects when slabs are split in temporal refinement. 

The second category are general classes that do not directly depend on the slab triangulation. 
Examples are various quadrature formulae, finite elements and FEValues classes.

### The purpose of the `TimeIteratorCollection`?
To simplify time-marching all spacetime list iterators and their respective collection 
can be added to an iterator collection. Any iterators registered with the collection will be incremented or decremented
together, which tidies up the time-marching loops. 

## Some details on slab assembly
----
In essence the assembly in space-time could be done by hand with double the amount of loops over 
elements, dofs and quadrature points. 
Especially the evaluation of space-time shape functions by multiplication of spatial and temporal 
shape function values is error prone and lacks a certain readabilty. 

### The `spacetime::FEValues` class
This class implements tensor-product finite element shape functions by handling all multiplications internally.
As an example we will look at $((\partial_t u_{kh},\varphi_{kh}))$ on a single space-time element. 


By hand the assembly in pseudo code for would be 

    for (spatial quadrature point qx in spatial quadrature formula){
    for (temporal quadrature point qt in spatial quadrature formula){
        for (spatial dof index i){
        for (spatial dof index j){
            for (temporal dof index ii){
            for (temporal dof index jj){
                spacetime index iii = i + n_dofs_space*ii;
                spacetime index jjj = j + n_dofs_space*ii;
                local_matrix(iii,jjj) += 
                    space_fe_values.shape_value(i,qx)*
                        time_fe_values.shape_value(ii,qt) *
                    space_fe_values.shape_value(j,qx)*
                        time_fe_values.shape_grad(jj,qt)[0]*
                    space_fe_values.JxW(qx)*
                        time_fe_values.JxW(qt);
            }}
        }}
    }}

In contrast the assembly with ideal.II is 

    for (spacetime quadrature point q in spacetime quadrature formula){
        for (spacetime dof index i){
            for (spacetime dof index j){
                local_matrix(i,j) +=
                    spacetime_fe_values.shape_value(i,q)*
                    spacetime_fe_values.shape_dt(j,q)*
                    spacetime_fe_values.JxW(q);
            }
        }
    }

For efficiency reasons and to simplify usage of the WorkStream class later on, 
there is no space-time cell/element class and there are still two element loops instead.
Consequently the FEValues object has two reinit functions i.e. `reinit_space` and `reinit_time`
and the possibly slower `reinit_space` only has to be called after the whole temporal column is assembled.

### The FEJumpValues class
This class provides `shape_value_plus` and `shape_value_minus` to assemble the jump terms.
As these are only part of a spatial inner product both the element and quadrature loops are only done
in space. However, the dof loops still need to be in space-time as Gauss-Legendre support points don't lie on the element boundary and so $u^+$ and $u^-$ are linear combinations of the spatial shape values at each temporal degree of freedom. 

## What is different compared to a classical stationary FEM Code? 
-----
In this small section we want to compare the sequence of function calls with the step-3 Poisson example from deal.II
In the stationary example the basic sequence in the main `run()` function is as follows
* `make_grid()` produces a uniformly refined unit square mesh
* `setup_system()` distributes the degrees of freedom on the given mesh and sets the appropriate sizes of vectors and matrices
* `assemble_system()` loops over all elements in the mesh, computes the local contributions to the matrix and right hand side and adds those to the global matrix $A$ and vector $b$.
* `solve()` solves the system $Au=b$ with conjugate gradients CG.
* `output_results()` writes $u$ into a vtk file that can be read by visualization tools like VisIt or Paraview

If you are familiar with time-stepping codes you know that we would add a loop over the time-steps around some of these functions. 
As we use slabs the time-marching will be done over those and our main function `run()` functions 
just has the two steps
* `make_grid()` to produce first a spatial unit square mesh and then propagate it through time in a `spacetime::fixed::Triangulation` 
* `do_time_marching()` which contains the loop over the `TimeIteratorCollection` 

Then inside this loops we call
* `setup_system_on_slab()` distribute space-time dofs on the given slab and set the sizes of the space-time vectors and matrices
* `assemble_system_on_slab()` As explained above iterate over all spatial elements and add their local contributions in space-time to the full matrix and right hand side
* `solve_system_on_slab()` solve the slab system with a direct solver. CG is not applicable as the matrix is not symmetrical due to the jump terms and temporal derivative.
* `output_results_on_slab()` Write the solution at each temporal dof into its own vtk file. 
* preparing the initial value for the next slab (i.e. extracting the final time dof values)


  
