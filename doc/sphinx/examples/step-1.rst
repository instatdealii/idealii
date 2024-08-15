.. _step-1:

************************************
Example 1: Solving the heat equation
************************************

.. contents::
    :local:

Introduction
============

In this tutorial we will look at how the basic concepts of `ideal.II` 
can be combined to solve a nonstationary PDE. 
This might repeat earlier definitions or explanations, 
but the goal is to see how everything will be put together 
to show a complete picture. The exemplary equation 
will again be the heat equation as it is a linear and scalar equation 
with well known terms and operators.

As before, we are interested in the solution on 
a space-time domain (cylinder), i.e. :math:`Q = \Omega\times I` 
with spatial domain :math:`\Omega\subset\mathbb{R}^d` 
and temporal domain :math:`I = (0,T)`. 
Given an initial solution :math:`u^0\in L^2(\Omega)`,
a force term :math:`f\in L^2(I,)` and a Dirichlet 
boundary function :math:`g\in L^2(I,)` the heat equation reads as:

.. maths-equation:: Heat equation in strong formulation

    The strong formulation reads as follows:

    .. math:: 

        \partial_t u - \Delta u &= f\text{ in }Q\\
        u &= g \text{ on }\partial\Omega\times I\\
        u &= u^0 \text{ on }\Omega\times\{0\}


Weak formulation of the heat equation
-------------------------------------

As a next step we will derive the weak formulation needed 
to do a finite element discretization.
Using the fully continuous function space 
:math:`X=X(I,H^1_0(\Omega))=\{v\in L^2(I,H^1_0(\Omega));\partial_t v\in L^2(I,H^1_0(\Omega)^*)\}`
we can multiply the strong form by test functions :math:`\varphi\in X`
and integrate over :math:`Q` to obtain:

.. maths-equation:: Heat equation in weak formulation

    Find :math:`u\in X+g`
    such that:

    .. math:: 

        \int\limits_0^T\int\limits_\Omega \varphi\partial_t u + 
        \nabla_x\varphi\nabla_x u\;\mathrm{d}x\;\mathrm{d}t
        &= \int\limits_0^T\int\limits_\Omega \varphi f  \;\mathrm{d}x\;\mathrm{d}t\; \forall\varphi\in X\\
        u(0,x) &= u^0(x) \text{ in }\Omega

Space-time discretization
-------------------------

While this equation can be solved analytically in some cases,
we are interested in approximately solving the equation 
on our computer. Since our space-time cylinder is a cartesian 
product we can construct function spaces by tensor products of 
spatial and temporal function spaces 
and space-time basis functions by multiplication (see :ref:`tp_hilbert_spaces`).
Therefore, we can split the discretization, starting with the temporal part.

Here, we want to use the discontinuous Galerkin method 
so we replace :math:`X(I,H^1_0(\Omega))` by the temporally
discontinuous space :math:`\widetilde{X}(\mathcal{T}_k,H^1_0(\Omega))`
depending on our temporal triangulation 
:math:`\mathcal{T}_k = \{I_1,I_2,\dots,I_M\}` with :math:`I_m=(t_{m-1},t_m)`.
To account for the discontinuity we introduce the jump terms
:math:`\varphi_m^+[u]_m=\varphi_m^+(u_m^+-u_m^-)` and 
:math:`\varphi_0^+[u]_0=\varphi_m^+(u_0^+-u^0)` and
split the temporal integral at the element faces.
With this we can now state the discontinuous weak formulation:

.. maths-equation:: Heat equation in discontinuous weak formulation

    Find :math:`u\in \widetilde{X}+g`
    such that:

    .. math:: 

        \sum\limits_{m=1}^{M}&\int\limits_{t_{m-1}}^{t_m}\int\limits_\Omega 
            \varphi\partial_t u + 
            \nabla_x\varphi\nabla_x u +
        \;\mathrm{d}x\;\mathrm{d}t
        +\sum\limits_{m=0}^{M}\int\limits_\Omega 
            \varphi_m^+[u]_m\;\mathrm{d}x\\
        = \sum\limits_{m=1}^{M}&\int\limits_{t_{m-1}}^{t_m}\int\limits_\Omega \varphi f  \;\mathrm{d}x\;\mathrm{d}t\; \forall\varphi\in X

The actual refinement is done in two steps.
We start by discretizing the temporal part into piecewise discontinuous 
Lagrange elements of order :math:`r` to obtain a time-discrete solution
:math:`u_k`. 
Afterwards we discretize :math:`\Omega` with a spatial mesh 
:math:`\mathcal{T}_h` and :math:`H^1_0(\Omega)` by continuous 
Lagrange elements of order :math:`s` on that mesh.
Then we have the fully discrete function space 
:math:`\widetilde{X}_{kh}^{r,s}(\mathcal{T}_k,\mathcal{T}_h)` 
and solution :math:`u_{kh}`. 

Finally, we also have to project the boundary function :math:`g(t,x)`
into a finite element representation :math:`\check{g}(t,x)\in\widetilde{X}_{kh}^{r,s}`.

Problem statement
-----------------
Finally, we want to state the problem 
description of the actual configuration 
we want to solve.
The spatial domain will be the unit square
:math:`\Omega=(0,1)^2` and the final time is :math:`T=1`.
This results in a unit cube space-time cylinder 
:math:`\Sigma=(0,1)^3`.
To be able to validate our implementation,
we will derive the right hand side 
form a simple manufactured solution

.. math::
    u_{\text{exact}}(t,x) = -(x^2-x)(y^2-y)t/4.

Inserting this into the strong form, we obtain

.. math::
    f(t,x) = (y^2-y)t/2+(x^2-x)t/2-(x^2-x)(y^2-y)/4

What is different compared to a classical stationary FEM code?
--------------------------------------------------------------

Here, we want to compare the sequence of function calls with 
the step-3 Poisson example from deal.II.
In the stationary example the basic sequence in the main ``run()`` function 
is as follows:

* ``make_grid()`` produces a uniformly refined unit square mesh
* ``setup_system()`` distributes the degrees of freedom on the given mesh and sets the appropriate sizes of vectors and matrices
* ``assemble_system()`` loops over all elements in the mesh, computes the local contributions to the matrix and right hand side and adds those to the global matrix $A$ and vector $b$.
* ``solve()`` solves the system $Au=b$ with conjugate gradients CG.
* ``output_results()`` writes $u$ into a vtk file that can be read by visualization tools like VisIt or Paraview


If you are familiar with time-stepping codes, 
you know that we would add a loop over the time-steps around some of these functions. 
As we use slabs, the time-marching will be done over those functions
and our main function ``run()``  just has the two steps

* ``make_grid()`` to produce first a spatial unit square mesh and then propagate it through time in a ``spacetime::fixed::Triangulation``
* ``do_time_marching()`` which contains the loop over the ``TimeIteratorCollection``

Then inside this time marching loop we call:

* ``setup_system_on_slab()`` distribute space-time dofs on the given slab and set the sizes of the space-time vectors and matrices
* ``assemble_system_on_slab()`` As explained above iterate over all spatial elements and add their local contributions in space-time to the full matrix and right hand side
* ``solve_system_on_slab()`` solve the slab system with a direct solver. CG is not applicable as the matrix is not symmetrical due to the jump terms and temporal derivative.
* ``output_results_on_slab()`` Write the solution at each temporal dof into its own vtk file. 
* preparing the initial value for the next slab (i.e. extracting the final time dof values)


The commented program
=====================

.. cpp-example:: ../../examples/step-1/step-1.cc

Results
=======
The first lines of output of the program look as follows:

.. code-block:: console
      
  *******Starting time-stepping*********
  Starting time-step (0,0.01]
  Number of degrees of freedom: 
  	  4225 (space) * 1 (time) = 4225
  Starting time-step (0.01,0.02]
  Number of degrees of freedom: 
  	  4225 (space) * 1 (time) = 4225

Note that this output of course depends on the number 
temporal elements :math:`M` of spatial 
refinements as set in ``make_grid()`` as well as the chosen 
finite elements.
For a dG(1) discretization in time the output would instead be:

.. code-block:: console
      
  *******Starting time-stepping*********
  Starting time-step (0,0.01]
  Number of degrees of freedom: 
  	  4225 (space) * 2 (time) = 8450
  Starting time-step (0.01,0.02]
  Number of degrees of freedom: 
  	  4225 (space) * 2 (time) = 8450

Note that we would also get the above output with dG(0) elements
and a single temporal refinement, which results in two temporal 
elements per slab.

The ``.vtk`` files generated during output can be opened by many 
visualization programs, including `Paraview <https://www.paraview.org/>`_
and `VisIt <https://visit-dav.github.io/visit-website/>`_.
Since the solution is time dependent we have used Paraview to generate 
the following video:

.. video:: ../_static/examples/step-1.ogv
    :loop:
    :height: 500
    :autoplay:

It shows the solution, i.e. the function :math:`(x^2-x)(y^2-y)/4` 
scaled by :math:`t`.

Possibilities for extensions
----------------------------

If you want to play around with this program here are a few suggestions:

* Change the geometry and mesh: We generated a square domain but deal.II's 
  `GridGenerator <https://dealii.org/current/doxygen/deal.II/namespaceGridGenerator.html>_`
  has quite a few other options. 
* Change the finite element orders: A big advantage of space-time finite 
  elements is the possibility to change the convergence order of the 
  temporal discretization by increasing the degree of the temporal elements.
  Note, that the approximate solution might seem wrong initially as it 
  'wiggles'  when going through the output files. 
  This is however correct. 
  What we see are the two different values at each inner temporal edge 
  due to the discontinuous Galerkin discretization.
* Observe convergence: We will discuss computing the :math:`L^2(Q)`-error
  in :ref:`step-3` where the given exact solution is much more challenging 
  in time compared to the linear behaviour of :math:`u`.
  This will allow us to study the performance of different 
  temporal and spatial element orders.
  Here, we could instead use ``dealii::VectorTools::compute_mean_value()``
  for the spatial solution on each temporal element and average them 
  over the temporal interval. 
  Integrating the exact solution over :math:`(0,T)\times\Omega` 
  and dividing by :math:`|\Omega|T` we get the expected value :math:`-T/288`
  to which the approximated mean value will converge.


The plain program
=================
    
.. cpp-example-plain:: ../../examples/step-1/step-1.cc
