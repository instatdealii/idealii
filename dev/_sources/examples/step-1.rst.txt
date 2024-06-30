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

As before we are interested in the solution on 
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
and space-time basis functions by multiplication (see TODO: TENSOR PRODUCT HILBERT SPACES).
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

    

.. .. maths-statement:: Theorem: Zeidler

..     Let evolution triple, get continuous embedding


The commented program
=====================

.. cpp-example:: ../../examples/step-1/step-1.cc

Results
=======

The plain program
=================
    
.. cpp-example-plain:: ../../examples/step-1/step-1.cc

