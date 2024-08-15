.. _examples:

***********************************
Example overview
***********************************

The tutorial examples expect that you have some familiarity 
with using `deal.II`. 
If that is not the case we recommend doing the 
tutorial examples 
`step-1 <https://dealii.org/current/doxygen/deal.II/step_1.html>`_,
`step-2 <https://dealii.org/current/doxygen/deal.II/step_2.html>`_,
and 
`step-3 <https://dealii.org/current/doxygen/deal.II/step_3.html>`_
first. 
If any tutorials require knowledge about some concepts of `deal.II`,
we will link to the respective tutorial steps in their introduction.

Tutorials listed by number (chronological)
------------------------------------------

- :doc:`step-1`
- :doc:`step-2`
- :doc:`step-3`
- :doc:`step-4`

Tutorial overview by number 
---------------------------


step-1: Heat equation
^^^^^^^^^^^^^^^^^^^^^
This step serves as an introduction to tensor-product 
space-time finite elements and the basic structure of the library. 
It is recommended to know the *deal.II* tutorial examples listed above

step-2: Stokes equation
^^^^^^^^^^^^^^^^^^^^^^^
This step discusses the handling of coupled problems with vector-valued
components. 
It also explains how time-dependent nonhomogeneous Dirichlet boundary 
conditions are set in *ideal.II*.

We recommend to know the *deal.II* tutorial 
`step-22 <https://dealii.org/current/doxygen/deal.II/step_22.html>`_ 
which solves the stationary version of the Stokes equation.

step-3: Heat equation with Trilinos+MPI distributed linear algebra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step shows how to parallelize your *ideal.II* code (step-1), 
especially the handling of the indices for distributed degrees of freedom.
Additionally, we vary the support points for the temporal discontinuous 
Galerkin elements and the changes needed to arrive at a correct result again.
To compare these support point choices we calculate the space-time 
:math:`L^2`-error for various refinement levels and finite element degrees.


We recommend familiarity with an MPI parallel *deal.II* tutorial,
e.g. `step-40 <https://dealii.org/current/doxygen/deal.II/step_40.html>`.

step-4: Navier-Stokes equation with distributed linear algebra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step extends step-2 by making it parallel and handling the nonlinear
convection term added by the Navier-Stokes equations.
It discusses Newton linearization of a nonlinear PDE 
and introduces calculation of point and boundary functionals.




    
