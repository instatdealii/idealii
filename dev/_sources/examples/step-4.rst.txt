.. _step-4:

**********************************************************
Example 4: Solving the Navier-Stokes equations in parallel
**********************************************************

.. contents::
    :local:

Introduction
============

In this tutorial we will look at how to solve a nonliner problem in 
*ideal.II*.

It extends the couples Stokes equation from :ref:`step-2` 
and solves the nonlinear Navier-Stokes equation.
For the linear algebra it builds upon :ref:`step-3` and uses 
the MPI parallel linear algebra provided through Trilinos as well.

The Navier-Stokes equation is again a coupled problem between a vector-valued velocity 
:math:`\bf u\in\mathbb{R}^d` and a scalar-valued pressure :math:`p\in\mathbb{R}`.
Compared to the previous Stokes equation we now take into account 
the effect of the fluid motion on the fluid itself through the 
nonlinear convection term :math:`\mathbf{u}\cdot\nabla\mathbf{u}`.

.. maths-equation:: Navier-Stokes equation in strong formulation

    The strong formulation reads as follows:

    .. math:: 

        \partial_t \mathbf{u} - \nu\Delta\mathbf{u} 
        +\mathbf{u}\cdot\nabla_x\mathbf{u}+\nabla_x p &= 0\text{ in }Q \\
        \nabla_x\cdot \mathbf{u} &= 0 \\
        \mathbf{u} &= \mathbf{g}_D \text{ on }\Gamma_D\times I\\
        \partial_n\mathbf{u} &= 0\text{ on }\Gamma_N\times I\\
        \mathbf{u} &= \mathbf{u}^0 \text{ on }\Omega\times\{0\}

    with kinematic viscosity :math:`\nu\in\mathbb{R}`.


Weak formulations of the Navier-Stokes equation
-----------------------------------------------

For the weak formulation we need choose the same function spaces 
as in for Stokes and obtain:

.. maths-equation:: Navier-Stokes equation in weak formulation

    Find :math:`\mathbf{u}\in\mathbf{X}+\mathbf{g}_D, p\in Y`
    such that:

    .. math:: 
         &(\partial_t \mathbf{u},\mathbf{v}) 
         + \nu(\nabla_x {\mathbf{u}},\nabla_x \mathbf{v}) 
         +(\mathbf{u}\cdot\nabla_x\mathbf{u},\mathbf{v})\\
         &- (p,\nabla_x\cdot \mathbf{v}) 
         + (\nabla_x\cdot\mathbf{u},q) = 0\\
        &\mathbf{u}(0,x) = \mathbf{u}^0(x) \text{ in }\Omega
        
As in step-1 we allow for discontinuities between two temporal elements
in the function space for :math:`\bf{u}` and obtain:

.. maths-equation:: Stokes equation in discontinuous weak formulation

    Find :math:`\mathbf{u}\in\widetilde{\mathbf{X}}+\mathbf{g}_D, p\in Y`
    such that:

    .. math:: 
         \sum\limits_{m=1}^M &(\partial_t \bf{u},\bf{v})_{I_m\times\Omega}
         + \nu(\nabla_x {\bf{u}},\nabla_x \bf{v})_{I_m\times\Omega}
         + (\mathbf{u}\cdot\nabla_x\mathbf{u},\mathbf{v})_{I_m\times\Omega}\\
         &- (p,\nabla_x\cdot \bf{v})_{I_m\times\Omega}
         +(\nabla_x\cdot\bf{u},q)_{I_m\times\Omega} \\
         \sum\limits_{m=0}^M &+ ([\mathbf{u}]_m,\mathbf{v}_m^+)_{I_m\times\Omega}= 0 
        
For the temporal discretization we do the same steps as before
and for the spatial discretization we have to use inf-sup stable element
combinations. The simplest of these is the Q2/Q1 Taylor-Hood element 
with biquadratic elements for the velocity and bilinear elements for 
the pressure.

Problem statement
-----------------

Now we actually solve the 2D-3 benchmark problem from [TurSchaBen1996]_. 
The problem describes laminar flow of a fluid through a channel with a cylindrical
obstacle, as shown in the following image:

.. image:: ../_static/examples/channel_domain.svg

For the pressure we prescribe homogeneous Neumann conditions 
on all  boundaries and for the velocity we prescribe the following:

* A parabolic velocity profile on the inflow :math:`\Gamma_\text{in}`
* no-slip,i.e. homogeneous Dirichlet conditon on the obstacle 
  :math:`\Gamma_\text{circle}` and channel walls :math:`\Gamma_\text{wall}`
* A homogeneous Neumann condition on the outflow :math:`\Gamma_\text{out}`

We scale the inflow condition by a sine functions along the 
temporal domain :math:`I=(0,8)` and obtain

.. math::
    \mathbf{u}_x &= \sin(\pi t/8)(6y(H-y))/(H^2)\\
    \mathbf{u}_y &= 0

with channel height :math:`H=0.41`.


Since this is a benchmark problem there are functional values we can compute to
compare the results to other software packages.

We are going to calculate the temporal curves of the pressure at the 
front and back of the obstacle, i.e. at :math:`p(t,(0.15,0.1))` and :math:`p(t,(0.15,0.2))`,
their difference and the drag and lift coefficients

.. math::

    C_D(t) = \frac{2}{U^2L}F_D(t),\quad C_D(t) = \frac{2}{U^2L}F_L(t).

For this exact configuration the factor is :math:`\frac{2}{H^2L}=20`.
The drag and lift forces :math:`F_D(t)` and :math:`F_L(t)` around the 
cylinder are defined by

.. math::

    F_D(t) = \int_{\Gamma_\text{circle}} (-p(t)I+
    \nu\nabla \mathbf{v}(t))\cdot \mathbf{n}\cdot \mathbf{e}_1\mathrm{d}s\\
    F_L(t) = \int_{\Gamma_\text{circle}} (-p(t)I+
    \nu\nabla \mathbf{v}(t))\cdot \mathbf{n}\cdot \mathbf{e}_2\mathrm{d}s


The commented program
=====================

.. cpp-example:: ../../examples/step-4/step-4.cc

Results
=======

As before we start with the animations of the resulting fields,
i.e. velocity magnitude and pressure, in the VTU files.

.. video:: ../_static/examples/step-4-velocity.ogv
    :loop:
    :height: 800
    :autoplay:


.. video:: ../_static/examples/step-4-pressure.ogv
    :loop:
    :height: 800
    :autoplay:

Comparison to benchmark results
-------------------------------

Finally, we want to compare our results to results from the 
finite element software 
`FEATFLOW <https://wwwold.mathematik.tu-dortmund.de/~featflow/en>`_
which are published on their 
`benchmark website <https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html>`_.

The *ideal.II* results have been obtained by runnning the R files in the example
folder. 

We can see that the  drag coefficient mostly depends on the 
spatial discretization as our results overlay each other 
until :math:`t\approx 4.2`. 
We can also see that :math:`dG(0)` produces a smoother curve
due to numerical dampening.

.. image:: ../_static/examples/NSE_results_idealii_vs_FEATFLOW_drag.svg

For the lift we can see the numerical dampening even more clearly
as the :math:`dG(0)` discretization does not produce 
the correct oscillations.
For the higher order discretizations we see that the 
oscillation frequency is close to the results of FEATFLOW 
with refinement level 2, which has a similar number of 
spatial DoFs.
We also see that the peaks are not matching, which 
might in part be because there is no temporal DoF close enough
to the peak point.

.. image:: ../_static/examples/NSE_results_idealii_vs_FEATFLOW_lift.svg

For the pressure we see a similar picture compared to the drag,
but now our results are almost completely overlaying the 
FEATFLOW level 2 results.
.. image:: ../_static/examples/NSE_results_idealii_vs_FEATFLOW_pressure.svg


The plain program
=================
    
.. cpp-example-plain:: ../../examples/step-4/step-4.cc



