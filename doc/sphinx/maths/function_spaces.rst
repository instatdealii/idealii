.. _function_spaces:

************************************
Space-time function spaces
************************************

.. contents::
    :local:

Introduction
============

The basis for the finite element method is the definition 
of weak solutions and weak formulations.
To construct them, we need the right function spaces 
for the test functions and the weak solution itself.
In this chapter we will look at the mathematical foundations 
of the space-time function spaces needed for 
the tensor product finite element approach.

For the finite subspaces needed to actually apply
the finite element method and obtain a linear equation 
system see the next chapter :ref:`discretization`

Basic notation 
==============




.. _tp_hilbert_spaces:

Tensor product Hilbert spaces
=============================

We will start by constructing Hilbert spaces 
on space-time cylinders, which are formed through a cartesian product,
i.e. :math:`\Sigma=I\times\Omega`.
For more details and proofs for the construction, see [PicardMcGhee2011]_.

In the following we will have two infinite dimensional Hilbert spaces :math:`H_a(I)`
and :math:`H_b(\Omega)` with their respective finite dimensional subspaces 
:math:`V_a \subset H_a` and :math:`V_b \subset H_b`.
If at least one space is finite then :math:`\otimes` will
denote the algebraic tensor product, i.e. :math:`V_a\otimes H_b`.
If both spaces are infinite, then :math:`\hat{\otimes}`
will denote the closure of the Hilbert space tensor product.

Following the notation of the spaces :math:`L^p(I,X)` 
and :math:`W^{1,p}(I,X)` in [Evans2010]_\ ,
we can define :math:`H_a(I,H_b(\Omega))`
as the space of :math:`H_a` functions over :math:`I`
with values in :math:`H_b(\Omega)`.
With this we can now state three important results for the 
construction of space-time functions.

.. maths-statement:: Proposition (1.2.27)

    :math:`H_a(I)\hat{\otimes} H_b(\Omega)` 
    is a Hilbert space and isometric to :math:`H_a(I,H_b(\Omega))`.
    
.. maths-statement:: Proposition (1.2.28)

    Let :math:`V_a(I)` be a finite subspace of :math:`H_a(I)`.
    Then, :math:`V_a(I)\otimes H_b(\Omega)\subset H_a\hat{\otimes}H_b` is a Hilbert space 
    and isometric to :math:`V_a(I,H_b(\Omega))`.

.. maths-statement:: Proposition (1.2.28)

    Additionally, let :math:`V_b(\Omega)` be a finite subspace of :math:`H_b(\Omega)`.
    Then, :math:`V_a(I)\otimes V_b(\Omega)\subset V_a\otimes H_b` is a Hilbert space 
    and isometric to :math:`V_a(I,V_b(\Omega))`.

Importantly, we can use the propositions to identify functions
:math:`f\in H_a(I,H_b(\Omega))` with the product of functions 
:math:`g\in H_a(I)` and :math:`h\in H_b(\Omega)`, i.e. 

.. math::
    f(t,x) = g(t)h(x).

Time-dependent Sobolev spaces
=============================

.. maths-statement:: Definition (Evolution triple)
    
    An **evolution triple**, also called **Gelfand triple**

    .. math::
        V\subseteq H \subseteq V^* 

    or **rigged Hilbert space** :math:`(H,V)` 
    has the following properties

    * :math:`V` is a real, separable and reflexive Banach space
    * :math:`H` is a real, separable Hilbert space
    * The embedding :math:`V\subseteq H` is continuous

In the following we will use the special case, where :math:`V` 
is also a Hilbert space. 
Based on such an evolution triple we can now define our
time-dependent Sobolev spaces

.. maths-statement:: Definition (Time-dependent Sobolev space)

    Let :math:`V(\Omega)` be a Hilbert space over :math:`\Omega` 
    which forms an evolution triple with :math:`L^2(\Omega)`
    and let :math:`V(\Omega)^*` be its dual space.
    Then, we can define the space 

    .. math::
        W(I,V(\Omega))\;\colon= \{v\in L^2(I,V(\Omega)); \partial_t v\in L^2(I,V^*(\Omega))\}

These spaces are also **Bochner spaces**, i.e. function spaces
with values in Banach spaces [Růžička2020]_.



References
============

.. [PicardMcGhee2011] Picard, L. and McGhee, D. *Partial Differential Equations*. De Gruyter, Berlin, New York, 2011.
.. [Evans2010] Evans, L. C. *Partial Differential Equations*, volume 19. American Mathematical Society, 2010.
.. [Růžička2020] Růžička. *Nichtlineare Funktionalanalysis* Springer Berlin Heidelberg, 2020



