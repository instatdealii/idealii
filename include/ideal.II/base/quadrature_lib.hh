// ---------------------------------------------------------------------
//
// Copyright (C) 2022 - 2023 by the ideal.II authors
//
// This file is part of the ideal.II library.
//
// The ideal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of ideal.II.
//
// ---------------------------------------------------------------------

#ifndef INCLUDE_IDEAL_II_BASE_QUADRATURE_LIB_HH_
#define INCLUDE_IDEAL_II_BASE_QUADRATURE_LIB_HH_
#include <ideal.II/base/spacetime_quadrature.hh>

#include <deal.II/base/quadrature_lib.h>

namespace idealii
{



  /**
   * The Gauss-Radau family of quadrature rules for numerical integration.
   *
   * This modification of the Gauss quadrature uses one of the two interval end
   * points as well. Being exact for polynomials of degree $2n-2$, this
   * formula is suboptimal by one degree.
   *
   * This formula is often used in the context of discontinuous Galerkin
   * discretizations of ODEs and the temporal part of PDEs.
   *
   * The quadrature points are the left interval end point plus the $n-1$
   * roots of the polynomial
   * \f[
   *   \frac{P_{n-1}(x)+P_n(x)}{1+x}
   * \f]
   * where $P_{n-1}$ and $P_n$ are Legendre polynomials.
   * The quadrature weights are
   * \f[
   *   w_0=\frac{2}{n^2}\quad\text{and}
   *   \quad w_i=\frac{1-x_i}{n^2(P_{n-1}(x_i))^2}\text{ for }i>0
   * \f]
   *
   * For the right Gauss-Radau formula the quadrature points are
   * $\tilde{x}_i=1-x_{n-i-1}$ and the weights are $\tilde{w}_i=w_{n-i-1}$,
   * with $(x_i,w_i)$ as quadrature points
   * and weights of the left Gauss-Radau formula.
   *
   * @see https://mathworld.wolfram.com/RadauQuadrature.html
   */
  template <int dim>
  class QGaussRadau : public dealii::Quadrature<dim>
  {
  public:
    /*
     * EndPoint is used to specify which of the two endpoints of the
     * unit interval is used as quadrature point
     */
    enum EndPoint
    {
      /**
       * Left end point.
       */
      left,
      /**
       * Right end point.
       */
      right
    };

    /// Generate a formula wit <tt>n</tt> quadrature points
    QGaussRadau(const unsigned int n, EndPoint end_point = QGaussRadau::left);
    /**
     * Move constructor. We cannot rely on the move constructor for
     * `Quadrature`, since it does not know about the additional member
     * `end_point` of this class.
     */
    QGaussRadau(QGaussRadau<dim> &&) noexcept = default;

  private:
    const EndPoint end_point;
  };

  /**
   * @brief 1D right box rule.
   *
   * The quadrature with generalized support point 1 and weight 1.
   * This produces a quadrature that is exact for constant polynomials.
   *
   * Although this is not necessarily desirable, using this formula for dG(0)
   * elements in time leads to the backward Euler method.
   *
   */
  class QRightBox : public QGaussRadau<1>
  {
  public:
    /**
     * @brief Default constructor.
     *
     * Constructs the quadrature formula with weight[0] = 1.0 and
     * quadrature_points[0] = dealii::Point<1>(1.0)
     */
    QRightBox();
  };

  /**
   * @brief 1D left box rule.
   *
   * The quadrature with generalized support point 0 and weight 1.
   * This produces a quadrature that is exact for constant polynomials.
   *
   * Although this is not necessarily desirable, using this formula for dG(0)
   * elements in time leads to the forward Euler method.
   */
  class QLeftBox : public QGaussRadau<1>
  {
  public:
    /**
     * @brief Default constructor.
     *
     * Constructs the quadrature formula with weight[0] = 1.0 and
     * quadrature_points[0] = dealii::Point<1>(0.0)
     */
    QLeftBox();
  };


  namespace spacetime
  {

    /**
     *	@brief A Gauss-Legende quadrature formula in space and time.
     *
     *
     */
    template <int dim>
    class QGauss : public Quadrature<dim>
    {
    public:
      /**
       * @brief Generate a Gauss-Legendre quadrature in space and time
       *
       * Exact for polynomials of degree 2*@p n_spatial-1 in space and 2*@p
       * n_temporal-1 in time.
       * @param n_spatial Number of spatial quadrature points (in each space direction)
       * @param n_temporal Number of temporal quadrature points
       */
      QGauss(unsigned int n_spatial, unsigned int n_temporal);
    };

    /**
     * @brief A Gauss-Legende quadrature formula in space and right box rule in time.
     *
     * Applying this quadrature to a dG finite element degree of 0 in time
     * results in the backward Euler method.
     */
    template <int dim>
    class QGaussRightBox : public Quadrature<dim>
    {
    public:
      /**
       * @brief Generate a Gauss-Legende quadrature in space and a right box rule
       * quadrature in time.
       *
       * Exact for polynomials of degree 2*@p n_spatial-1 in space and 0 in
       * time.
       * @param n_spatial Number of spatial quadrature points (in each space direction)
       */
      QGaussRightBox(unsigned int n_spatial);
    };
    /**
     * @brief A Gauss-Legende quadrature formula in space and left box rule in time.
     *
     * Applying this quadrature to a dG finite element degree of 0 in time
     * results in the forward Euler method.
     */
    template <int dim>
    class QGaussLeftBox : public Quadrature<dim>
    {
    public:
      /**
       * @brief Generate a Gauss-Legende quadrature in space and a left box rule
       * quadrature in time.
       *
       * Exact for polynomials of degree 2*@p n_spatial-1 in space and 0 in
       * time.
       * @param n_spatial Number of spatial quadrature points (in each space direction)
       */
      QGaussLeftBox(unsigned int n_spatial);
    };
  } // namespace spacetime
} // namespace idealii

#endif /* INCLUDE_IDEAL_II_BASE_QUADRATURE_LIB_HH_ */
