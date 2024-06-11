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

#include <ideal.II/base/quadrature_lib.hh>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <cmath>
#include <vector>
namespace idealii
{
  namespace internal::QGaussRadau
  {
    // Implements lookup table after affine transformation to [0,1].
    //
    // Analytical values for [-1,1] and n < 4 listed on
    // https://mathworld.wolfram.com/RadauQuadrature.html
    // Values for n > 3 calculated with the Julia Package
    // FastGaussQuadrature.jl
    // https://github.com/JuliaApproximation/FastGaussQuadrature.jl
    //
    std::vector<double>
    get_left_quadrature_points(const unsigned int n)
    {
      std::vector<double> q_points(n);
      switch (n)
        {
          case 1:
            q_points[0] = 0.;
            break;
          case 2:
            q_points[0] = 0.;
            q_points[1] = 2. / 3.;
            break;
          case 3:
            q_points[0] = 0.;
            q_points[1] = (6. - std::sqrt(6)) * 0.1;
            q_points[2] = (6. + std::sqrt(6)) * 0.1;
            break;
          case 4:
            q_points[0] = 0.000000000000000000;
            q_points[1] = 0.212340538239152943;
            q_points[2] = 0.590533135559265343;
            q_points[3] = 0.911412040487296071;
            break;
          case 5:
            q_points[0] = 0.000000000000000000;
            q_points[1] = 0.139759864343780571;
            q_points[2] = 0.416409567631083166;
            q_points[3] = 0.723156986361876197;
            q_points[4] = 0.942895803885482331;
            break;
          case 6:
            q_points[0] = 0.000000000000000000;
            q_points[1] = 0.098535085798826416;
            q_points[2] = 0.304535726646363913;
            q_points[3] = 0.562025189752613841;
            q_points[4] = 0.801986582126391845;
            q_points[5] = 0.960190142948531222;
            break;
          case 7:
            q_points[0] = 0.000000000000000000;
            q_points[1] = 0.073054328680258851;
            q_points[2] = 0.230766137969945495;
            q_points[3] = 0.441328481228449865;
            q_points[4] = 0.663015309718845702;
            q_points[5] = 0.851921400331515644;
            q_points[6] = 0.970683572840215114;
            break;
          case 8:
            q_points[0] = 0.000000000000000000;
            q_points[1] = 0.056262560536922135;
            q_points[2] = 0.180240691736892389;
            q_points[3] = 0.352624717113169672;
            q_points[4] = 0.547153626330555420;
            q_points[5] = 0.734210177215410598;
            q_points[6] = 0.885320946839095790;
            q_points[7] = 0.977520613561287499;
            break;
          default:
            Assert(false, dealii::StandardExceptions::ExcNotImplemented());
            break;
        }
      return q_points;
    }

    std::vector<double>
    get_quadrature_points(const unsigned int                  n,
                          ::idealii::QGaussRadau<1>::EndPoint end_point)
    {
      std::vector<double> left_points = get_left_quadrature_points(n);
      switch (end_point)
        {
          case ::idealii::QGaussRadau<1>::left:
            {
              return left_points;
            }
          case ::idealii::QGaussRadau<1>::right:
            {
              std::vector<double> points(n);
              for (unsigned int i = 0; i < n; ++i)
                {
                  points[n - i - 1] = 1. - left_points[i];
                }
              return points;
            }
          default:
            {
              Assert(false,
                     dealii::ExcMessage(
                       "This constructor can only be called with either "
                       "QGaussRadau::left or QGaussRadau::right as "
                       "second argument."));
              return {};
            }
        }
    }

    // Implements lookup table after affine transformation to [0,1].
    //
    // Analytical values for [-1,1] and n < 4 listed on
    // https://mathworld.wolfram.com/RadauQuadrature.html
    // Values for n > 3 calculated with the Julia Package
    // FastGaussQuadrature.jl
    // https://github.com/JuliaApproximation/FastGaussQuadrature.jl
    //
    std::vector<double>
    get_left_quadrature_weights(const unsigned int n)
    {
      std::vector<double> weights(n);
      switch (n)
        {
          case 1:
            weights[0] = 1.;
            break;
          case 2:
            weights[0] = 0.25;
            weights[1] = 0.75;
            break;
          case 3:
            weights[0] = 1. / 9.;
            weights[1] = (16. + std::sqrt(6)) / 36.;
            weights[2] = (16. - std::sqrt(6)) / 36.;
            break;
          case 4:
            weights[0] = 0.062500000000000000;
            weights[1] = 0.328844319980059696;
            weights[2] = 0.388193468843171852;
            weights[3] = 0.220462211176768369;
            break;
          case 5:
            weights[0] = 0.040000000000000001;
            weights[1] = 0.223103901083570894;
            weights[2] = 0.311826522975741427;
            weights[3] = 0.281356015149462124;
            weights[4] = 0.143713560791225797;
            break;
          case 6:
            weights[0] = 0.027777777777777776;
            weights[1] = 0.159820376610255471;
            weights[2] = 0.242693594234484888;
            weights[3] = 0.260463391594787597;
            weights[4] = 0.208450667155953895;
            weights[5] = 0.100794192626740456;
            break;
          case 7:
            weights[0] = 0.020408163265306121;
            weights[1] = 0.119613744612656100;
            weights[2] = 0.190474936822115581;
            weights[3] = 0.223554914507283209;
            weights[4] = 0.212351889502977870;
            weights[5] = 0.159102115733650767;
            weights[6] = 0.074494235556010341;
            break;
          case 8:
            weights[0] = 0.015625000000000000;
            weights[1] = 0.092679077401489660;
            weights[2] = 0.152065310323392683;
            weights[3] = 0.188258772694559262;
            weights[4] = 0.195786083726246729;
            weights[5] = 0.173507397817250691;
            weights[6] = 0.124823950664932445;
            weights[7] = 0.057254407372128648;
            break;
          default:
            Assert(false, dealii::StandardExceptions::ExcNotImplemented());
            break;
        }
      return weights;
    }
    std::vector<double>
    get_quadrature_weights(const unsigned int                        n,
                           const ::idealii::QGaussRadau<1>::EndPoint end_point)
    {
      std::vector<double> left_weights = get_left_quadrature_weights(n);
      switch (end_point)
        {
          case ::idealii::QGaussRadau<1>::EndPoint::left:
            return left_weights;
          case ::idealii::QGaussRadau<1>::EndPoint::right:
            {
              std::vector<double> weights(n);
              for (unsigned int i = 0; i < n; ++i)
                {
                  weights[n - i - 1] = left_weights[i];
                }
              return weights;
            }
          default:
            Assert(false,
                   dealii::ExcMessage(
                     "This constructor can only be called with either "
                     "QGaussRadau::EndPoint::left or "
                     "QGaussRadau::EndPoint::right as second argument."));
            return {};
        }
    }
  } // namespace internal::QGaussRadau

#ifndef DOXYGEN
  template <>
  QGaussRadau<1>::QGaussRadau(const unsigned int n, EndPoint end_point)
    : dealii::Quadrature<1>(n)
    , end_point(end_point)
  {
    Assert(n > 0,
           dealii::ExcMessage("Need at least one point for quadrature rules"));
    std::vector<double> p =
      internal::QGaussRadau::get_quadrature_points(n, end_point);
    std::vector<double> w =
      internal::QGaussRadau::get_quadrature_weights(n, end_point);

    for (unsigned int i = 0; i < this->size(); ++i)
      {
        this->quadrature_points[i] = dealii::Point<1>(p[i]);
        this->weights[i]           = w[i];
      }
  }
#endif

  template <int dim>
  QGaussRadau<dim>::QGaussRadau(const unsigned int n, EndPoint end_point)
    : dealii::Quadrature<dim>(
        QGaussRadau<1>(n, static_cast<QGaussRadau<1>::EndPoint>(end_point)))
    , end_point(end_point)
  {}

  QRightBox::QRightBox()
    : QGaussRadau<1>(1, QGaussRadau<1>::EndPoint::right)
  {}

  QLeftBox::QLeftBox()
    : QGaussRadau<1>(1, QGaussRadau<1>::EndPoint::right)
  {}

  namespace spacetime
  {
    template <int dim>
    QGauss<dim>::QGauss(unsigned int n_spatial, unsigned int n_temporal)
      : Quadrature<dim>(std::make_shared<dealii::QGauss<dim>>(n_spatial),
                        std::make_shared<dealii::QGauss<1>>(n_temporal))
    {}

    template <int dim>
    QGaussRightBox<dim>::QGaussRightBox(unsigned int n_spatial)
      : Quadrature<dim>(std::make_shared<dealii::QGauss<dim>>(n_spatial),
                        std::make_shared<QRightBox>())
    {}

    template <int dim>
    QGaussLeftBox<dim>::QGaussLeftBox(unsigned int n_spatial)
      : Quadrature<dim>(std::make_shared<dealii::QGauss<dim>>(n_spatial),
                        std::make_shared<QLeftBox>())
    {}
  } // namespace spacetime
} // namespace idealii

#include "quadrature_lib.inst"
