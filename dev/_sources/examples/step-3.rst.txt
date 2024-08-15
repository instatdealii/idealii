.. _step-3:

************************************************
Example 3: Solving the heat equation in parallel
************************************************

.. contents::
    :local:

Introduction
============

This problem extends :ref:`step-1` in a few meaningful ways:

* The exact solution and right hand side are more challenging compared to a 
  solution that is linear in time and quadratic in space.
* We change the support points for the temporal discretization.
* The linear algebra is now distributed over multiple MPI processes by using 
  Trilinos. We will give a short overview over the important concepts, 
  but for a good in-depth explanation of MPI-parallel simulations we highly 
  recommend the deal.II ressources 
  `step-40 <https://dealii.org/current/doxygen/deal.II/step_40.html>`_
  ,which solves the Poisson or stationary heat equation, and the
  overview page `Parallel computing with multiple processors using distributed memory <https://dealii.org/current/doxygen/deal.II/group__distributed.html>`_.
* Calculation of the space-time :math:`L^2`-error and subsequent convergence 
  studies.
* Providing a command line interface to set simulation parameters for use in 
  a script based batch-processing. 
  This is used to run all the simulations for the convergence studies.

Support points for discontinuous elements
-----------------------------------------
For continuous Lagrange finite elements the outermost DoFs are always placed 
on the boundary. That way they can be combined with the DoFs of 
neighboring elements as they have to have the same values due to the continuity.

For discontinuous elements however, this is no longer necessary.
In the previous examples we have used the default of ``DGFiniteElement`` 
which are support points at the locations of Gauss-Lobatto quadrature points.
This quadrature rule is defined by requiring the first and last quadrature 
point to be on the boundary of the interval, like in the continuous case.

Additionally, *ideal.II* offers the following three values for ``support_type``

* Gauss(-Legendre) with no DoFs on the element boundary
* left Gauss-Radau with only the first DoF on the left boundary
* right Gauss-Radau with only the last DoF on the right boundary

These may provide better approximations as we can see in the results.
However, this introduces some important differences that have to be taken into
account

* The final value needed for the initial jump of the following slab might not be
  on a temporal DoF, so we have to use ``extract_subvector_at_time_point`` 
  instead of extracting the final subvector of the slab
* At least one of values needed for the jump values is no longer on a DoF and 
  we get a larger sparsity pattern. This is handled internally by  
  ``make_upwind_sparsity_pattern()`` so you only need to be aware of it.
* The output is no longer at the temporal grid points when using 
  ``extract_subvector_at_time_dof`` there. In the future this will be addressed
  by a space-time version of ``DataOutput``.

Concepts around distributed computing
-------------------------------------

We start by partitioning the spatial mesh and distributing it to 
each MPI rank/process. 
The Triangulation in ``parallel::distributed`` provides this functionality.
Then, each spatial element belongs to exactly one rank. 
Additionally, each spatial DoF is owned by only one rank. 
As a result a DoF can belong to a different rank but is also part of 
an owned element and will be needed for assembly. 
Therefore, there are two sets of DoF indices:

* The locally owned DoFs actually owned by the current rank
* The locally relevant DoFs which are all DoFs belonging to a locally owned 
  element, i.e. the locally owned DoFs plus so called 'ghost' DoFs.

These index sets will be used to initialize distributed vectors and matrices.
Note that only vectors initialized with the smaller locally owned set can be 
written into directly to avoid overriding a value on multiple MPI processes.

For the space-time index sets we take the spatial sets obtained from the 
``DoFHandler`` and shift them by the local temporal DoF index 
times the total number of spatial DoFs. 
So for a spatial mesh with 8 DoFs of which the indices :math:`(0,3,5,7)` 
are locally owned and a temporal mesh with 3 DoFs, the space-time index set 
would be :math:`(0,3,5,7,8,11,13,15,16,19,21,23)`.
PETSc does not allow for the specification of such an index set, 
so only Trilinos is supported in *ideal.II*.

Since some vectors, like the initial value, are defined on a single time point,
we need to know four index sets in total, two on the spatial and two on the 
space-time mesh.

Apart from that, all changes from step-1 to step-3 regarding distributed 
computing are standard changes you would also do in a stationary problem.
Some of these are 

* Calling ``MPI_InitFinalize`` in ``main()``.
* Knowing and saving the MPI Communicator.
* Changing the triangulation and linear algebra objects to distributed versions.
* Storing and using index sets during setup .
* Communicating between different locally owned and locally relevant vectors
* Compressing the matrix and vectors after assembly to communicate off-processor
  entries.
* Using parallel output functions from ``dealii::DataOut``


The commented program
=====================

.. cpp-example:: ../../examples/step-3/step-3.cc

Results
=======

As in the previous steps, we start with a video of the solution:

.. video:: ../_static/examples/step-3.ogv
    :loop:
    :height: 500
    :autoplay:

The ``.vtu`` files used to produce this animation have again been obtained by
running the program with default parameters. 

Convergence studies
-------------------

As :math:`u` is nonlinear we can use it to study the convergence behaviour of
the different discretizations, namely dG(0) to dG(2) with bilinear Q1 elements 
in space as well as dG(2) with biquadratic elements.
To get a better understanding we will look at uniform refinement only in time 
or space with a sufficently small mesh in the other dimensions
as well as uniform space-time refinement.

To make the following results easier to produce we provide to scripts in 
the `R language <https://www.r-project.org/>`_.
Both assume that you have built the executable `out-of-source`
by calling ``cmake -S. -Bbuild`` in the configure stage.
We recommend setting the build type to reduce with 
``cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release`` to get the best performance.
Depending on your system the simulations might still run a few hours,
The first, ``run_simuatlions.R`` starts by defining some convenience
functions.

* ``simulationcall`` for constructing the calls to ``mpirun``, if you want/have 
  to use more/less cores for the simulations you can change ``num_mpi_cores``,
* ``runsim`` for running through a set of spatial and temporal refinement 
  parameters and saving their respective results in ``.csv`` files,
* ``*_refinement`` for constructing the parameter sets to pass to ``runsim`` 
  depending on the type of refinement.

Finally, the script calls these refinement functions for the different 
discretization choices with Gauss-Lobatto support points.
As well as k-refinement for at least piecewise linear temporal 
elements for all support types to compare them.

h-refinement
^^^^^^^^^^^^

All (spatial) h-refinement runs start at the same spatial refinement level (4),
but have different temporal meshes such that the :math:`L^2`-errors are similar
enough to fit in the same plot.

As we can see from the following image, all refinements with bilinear elements
start parallel to each other. This is of course expected as the spatial 
convergence order does not depend on the temporal discretization.
However, we see that the curve for dG(0) elements stagnates. 
This is due to the fact that the overall :math:`L^2`-error is composed of a 
spatial and a temporal part. Since we do not refine the temporal mesh,
the temporal error dominates for dG(0) and the overall error converges to that 
part. 

.. image:: ../_static/examples/h_convergence_step-3.svg

k-refinement
^^^^^^^^^^^^
For the (temporal) k-refinement, all runs start with 10 temporal elements.
The spatial refinement is chosen in a way that all runs have the same amount of 
spatial DoFs, i.e. 7 refinement steps for bilinear and 6 for biquadratic
elements. 
We can again see that elements of the same order exhibit the same 
overall convergence behaviour. Here, due to the number of spatial DoFs being 
identical, the curves for dG(2) are even overlapping in the first few steps.
In this figure we can see the dominance of the non-refined error part even more 
clearly, as the dG(1) and dG(2) curve for bilinear elements converge to the same 
value. As the dG(0) curve is still above that error we don't yet see a stagnation.
Additionally, we see that the curve for the biquadratic elements also stagnates, 
but to a lower value. 

.. image:: ../_static/examples/k_convergence_step-3.svg

kh-refinement
^^^^^^^^^^^^^
For the uniform (space-time) kh-refinement we again start at 4 spatial 
refinement levels for all element combinations and use different initial 
temporal meshes to plot the curves close to each other.  
We can see three different convergence orders and we can see 
that the overall convergence is determined by the minimum of the 
spatial and temporal convergence order as a dG(2) discretization for 
bilinear elements produces about the same errors, but with :math:`3/2` 
of the amount of unknowns. More precisely, the relative error 
between cG(1)dG(1) and cG(1)dG(2) is around 1%  on the same space-time mesh.

To conclude, we see that matching the temporal and spatial element orders is 
important for achieving the optimal convergence. 
Luckily, due to the construction of the tensor product elements it is also easy
to match the orders. 

.. image:: ../_static/examples/kh_convergence_step-3.svg

Comparison of support types
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Finally, we have run the k-refinement studies for the different choices 
of support points and compared the results to the Gauss-Lobatto runs.
The tables below give the factor between the respective support point 
results and the Lobatto results in percent, 
i.e. :math:`100*L^2_{\text{other}}/L^2_{\text{Lobatto}`.
We can see that for an increasing number of temporal elements 
and for higher order discretizations, 
the choice of support points gets less important for this particular problem.
However, we see that the left Gauss-Radau rule consistently yields 
the lowest :math:`L^2`-error, followed by Gauss-Legendre and 
the right Gauss-Radau rule.
When keeping in mind that the left Gauss-Radau rule also has 
the second smallest sparsity pattern (after Gauss-Lobatto),
it makes it clear that choosing a different support type than Gauss-Lobatto
might be worthwhile depending on the problem.

.. list-table:: cG(1)dG(1)
    :widths: 25 25 25 25
    :header-rows: 1

    * - #DoFs
      - RadauLeft
      - Legendre
      - RadauRight

    * - 332820
      - 96.62%
      - 97.04%
      - 98.52%

    * - 665640 
      - 96.86%
      - 97.62%
      - 99.03%
    
    * - 1331280 
      - 97.37%
      - 97.98%
      - 99.02%
    
    * - 2662560 
      - 97.87%
      - 98.20%
      - 98.77%

.. list-table:: cG(1)dG(2)
    :widths: 25 25 25 25
    :header-rows: 1

    * - #DoFs
      - RadauLeft
      - Legendre
      - RadauRight

    * - 499230
      - 99.22%
      - 99.30%
      - 99.85%

    * - 998460 
      - 99.25%
      - 99.31%
      - 99.71%
    
    * - 1996920 
      - 99.79%
      - 99.81%
      - 99.91%
    
    * - 3993840 
      - 99.75%
      - 99.79%
      - 99.99%

.. list-table:: cG(2)dG(2)
    :widths: 25 25 25 25
    :header-rows: 1

    * - #DoFs
      - RadauLeft
      - Legendre
      - RadauRight

    * - 499230
      - 99.22%
      - 99.30%
      - 99.86%

    * - 998460 
      - 99.23%
      - 99.29%
      - 99.72%
    
    * - 1996920 
      - 99.32%
      - 99.42%
      - 99.72%
    
    * - 3993840 
      - 99.75%
      - 99.80%
      - 99.92%





The plain program
=================
    
.. cpp-example-plain:: ../../examples/step-3/step-3.cc


References
============

.. [Hartmann1998] Hartmann, R. *A- posteriori Fehlerschätzung und adaptive Schrittweiten- und Ortsgittersteuerung bei Galerkin-Verfahren für die Wärmeleitungsgleichung* Diploma Thesis, University of Heidelberg, 1998