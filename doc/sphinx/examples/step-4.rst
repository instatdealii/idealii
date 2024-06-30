.. _step-4:

**********************************************************
Example 4: Solving the Navier-Stokes equations in parallel
**********************************************************

.. contents::
    :local:

Introduction
============

.. Weak formulation of the heat equation
.. -------------------------------------
.. In this example we will solve the heat equation on 
.. a space-time domain (cylinder), i.e. :math:`Q = \Omega\times I` 
.. with spatial domain :math:`\Omega\subset\mathbb{R}^d` 
.. and temporal domain :math:`I = (0,T)`.

.. .. maths-equation:: Heat equation in strong formulation

..     The strong formulation reads as follows:

..     .. math:: 

..         \partial_t u - \Delta u &= f\text{ in }Q\\
..         u &= g \text{ on }\partial\Omega\times I\\
..         u &= u^0 \text{ on }\Omega\times\{0\}

.. Using the fully continuous function space 
.. :math:`X=\{v\in L^2(I,H^1_0(\Omega));\partial_t v\in L^2(I,H^1_0(Omega)^*)\}`
.. we can multiply the strong form by test functions :math:`\varphi\in X`
.. and integrate over :math:`Q` to obtain:

.. .. maths-equation:: Heat equation in weak formulation

..     Find :math:`u\in X+g`
..     such that:

..     .. math:: 

..         \int\limits_0^T\int\limits_\Omega \partial_t u\varphi + 
..         \nabla_x u\nabla_x\varphi \mathrm{d}x\mathrm{d}t
..         &= \int\limits_0^T\int\limits_\Omega f\varphi  \mathrm{d}x\mathrm{d}t \forall\varphi\in X\\
..         u(0,x) &= u^0(x) \text{ in }\Omega


.. .. maths-statement:: Theorem: Zeidler

..     Let evolution triple, get continuous embedding


The commented program
=====================

.. cpp-example:: ../../examples/step-4/step-4.cc

Results
=======

The plain program
=================
    
.. cpp-example-plain:: ../../examples/step-4/step-4.cc

