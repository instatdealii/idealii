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

# Example 3: Solving the heat equation in parallel 

## Mathematical background
-----------

We solve the same equation as in step 1.
The two main differences are a different exact solution introduced 
in the [diploma thesis](https://ganymed.math.uni-heidelberg.de/~hartmann/publications/1998/Har98_diploma.ps.gz) 
of Ralf Hartmann
and in the use of Gauss-Legendre support points for the discontinuous temporal finite elements

### Support points for discontinuous elements
As we don't require global continuity in time, the basis functions are only
supported locally on each temporal interval and independent of the other intervals.
Additionally, the support points no longer need to be on the boundary of the interval
to ensure continuity.

	Note:
	If you changed the default temporal degree from 1 to 0 in one of the previous steps
	this was already used implicitly as the degree of freedom is placed at the interval midpoint.

This allows us to place the degrees of freedom, i.e. the support points of the basis
functions, anywhere within the interval.
Of course, completely arbitrary support points might produce unexpected results and 
would be unwise. 
The good choice of support points for interpolatory basis functions is however the 
same problem as in quadrature formulae.
Starting from closed Newton-Cotes formulae with support points on the integral bounds,
better formulae are obtained by placing the points at the roots of Legendre polynomials
yielding Gauss-Legendre formulae.
And this is precisely what we can do here.
By setting the `support_type` to `Legendre` the temporal basis functions are constructed
with the support points of `dealii::QGauss` quadrature formulae.
The default type is `Lobatto` which uses `dealii::QGaussLobatto` for $r>0$ and places 
two support points at the interval ends. 

However, this introduces the difficulty that the initial and final value of a 
space-time finite element function on a single interval $(t_{m-1},t_m)$ are no longer just
subvectors of the space-time solution vector. 
Instead the resulting function has to be evaluated by a linear combination of all 
subvectors. The most direct consequence of this is on the jump values.
This has an impact at multiple places
1. In the `time_marching` function we now have to call `extract_subvector_at_time_point`
2. The sparsity pattern for multi interval slabs is now larger as all temporal dofs
of two neighboring intervals couple
3. The output is no longer at the temporal grid points when using `extract_subvector_at_time_dof`

3 will be addressed in ideal.II later on when the DataOutput functionality is implemented
and no longer handled manually in the example steps   
 

## Technical details on MPI parallelism in ideal.II
------

### Space-time indexing
The main challenge is the construction of space-time index sets for locally owned
and locally relevant (owned+ghost) degrees of freedom.
As in the sequential setting, the indices of the second temporal degree of freedom
are shifted by the number of spatial degrees of freedom (similar to the second row 
of a matrix when saved as a vector). 

This is the main reason why only Trilinos is supported as PetSc requires 
all indices belonging to a vector to be in a contiguous range without any gaps.

In practice you only need to be aware of this and avoid mixups between the 
spatial indexing needed for the initial value and transfer between slabs 
and the slab index sets spanning multiple dofs and possibly intervals.

### Other details
Apart from the index sets the changes from example 1 to the parallel version are
basically standard changes when extending a code to MPI parallelization.
Some of the important changes are
- Calling `MPI_InitFinalize` in the main function
- Knowing and saving the MPI Communicator
- Changing the triangulation to be `parallel::distributed` and vectors to be Trilinos vectors.
- owned and relevant sets of vectors and communication between the two versions
- Compressing the matrix after assembly, i.e. communcation between the processors
- Using parallel output functions from `dealii::DataOut`
