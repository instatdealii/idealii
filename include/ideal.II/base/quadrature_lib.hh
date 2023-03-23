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
     * @brief 1D right box rule.
     *
     * The quadrature with generalized support point 1 and weight 1.
     * This produces a quadrature that is exact for constant polynomials.
     *
     * Although this is not necessarily desirable, using this formula for dG(0)
     * elements in time leads to the backward Euler method.
     *
     */
    class QRightBox : public dealii::Quadrature<1>
    {
    public:
        /**
         * @brief Default constructor.
         *
         * Constructs the quadrature formula with weight[0] = 1.0 and
         * quadrature_points[0] = dealii::Point<1>(1.0)
         */
        QRightBox ();
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
    class QLeftBox : public dealii::Quadrature<1>
    {
    public:
        /**
         * @brief Default constructor.
         *
         * Constructs the quadrature formula with weight[0] = 1.0 and
         * quadrature_points[0] = dealii::Point<1>(0.0)
         */
        QLeftBox ();
    };

    namespace spacetime
    {

        /**
         *	@brief A Gauss-Legende quadrature formula in space and time.
         *
         *
         */
        template<int dim>
        class QGauss : public Quadrature<dim>
        {
        public:
            /**
             * @brief Generate a Gauss-Legendre quadrature in space and time
             *
             * Exact for polynomials of degree 2*@p n_spatial-1 in space and 2*@p n_temporal-1 in time.
             * @param n_spatial Number of spatial quadrature points (in each space direction)
             * @param n_temporal Number of temporal quadrature points
             */
            QGauss ( unsigned int n_spatial , unsigned int n_temporal );
        };

        /**
         * @brief A Gauss-Legende quadrature formula in space and right box rule in time.
         *
         * Applying this quadrature to a dG finite element degree of 0 in time
         * results in the backward Euler method.
         */
        template<int dim>
        class QGaussRightBox : public Quadrature<dim>
        {
        public:
            /**
             * @brief Generate a Gauss-Legende quadrature in space and a right box rule
             * quadrature in time.
             *
             * Exact for polynomials of degree 2*@p n_spatial-1 in space and 0 in time.
             * @param n_spatial Number of spatial quadrature points (in each space direction)
             */
            QGaussRightBox ( unsigned int n_spatial );
        };
        /**
         * @brief A Gauss-Legende quadrature formula in space and left box rule in time.
         *
         * Applying this quadrature to a dG finite element degree of 0 in time
         * results in the forward Euler method.
         */
        template<int dim>
        class QGaussLeftBox : public Quadrature<dim>
        {
        public:
            /**
             * @brief Generate a Gauss-Legende quadrature in space and a left box rule
             * quadrature in time.
             *
             * Exact for polynomials of degree 2*@p n_spatial-1 in space and 0 in time.
             * @param n_spatial Number of spatial quadrature points (in each space direction)
             */
            QGaussLeftBox ( unsigned int n_spatial );
        };
    }
}

#endif /* INCLUDE_IDEAL_II_BASE_QUADRATURE_LIB_HH_ */
