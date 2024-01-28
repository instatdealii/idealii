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

#include <cmath>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <ideal.II/base/quadrature_lib.hh>
#include <deal.II/base/quadrature_lib.h>
#include <vector>
namespace idealii
{
    namespace internal::QGaussRadau
    {
        //Implements lookup table as listed on
        //https://mathworld.wolfram.com/RadauQuadrature.html
        //but affine transformation from [-1,1] to [0,1]
        std::vector<double>
        get_left_quadrature_points(const unsigned int n)
        {
            std::vector<double> q_points(n);
            switch(n)
            {
                case 1:
                    q_points[0] = 0.;
                    break;
                case 2:
                    q_points[0] = 0.;
                    q_points[1] = 2./3.;
                    break;
                case 3:
                    q_points[0] = 0.;
                    q_points[1] = (6.-std::sqrt(6))*0.1;
                    q_points[2] = (6.+std::sqrt(6))*0.1;
                    break;
                default:
                    Assert(false, dealii::StandardExceptions::ExcNotImplemented());
                    break;
            }
            return q_points;
        }

        std::vector<double>
        get_quadrature_points(const unsigned int                  n,
                              ::idealii::QGaussRadau<1>::EndPoint ep)
        {
            std::vector<double> points(n);
            std::vector<double> left_points = get_left_quadrature_points(n);
            switch(ep)
            {
                case ::idealii::QGaussRadau<1>::left:
                {
                    return left_points;
                }
                case ::idealii::QGaussRadau<1>::right:
                {
                    for (unsigned int i = 0 ; i < n ; ++i){
                        points[i] = 1.-left_points[i];
                    }
                    return points;
                }
                default:
                {
                    Assert(
                        false,
                        dealii::ExcMessage(
                          "This constructor can only be called with either "
                          "QGaussRadau::left or QGaussRadau::right as "
                          "second argument."  
                        ));
                }
            }
        }

        //Implements lookup table as listed on
        //https://mathworld.wolfram.com/RadauQuadrature.html
        //but affine transformation from [-1,1] to [0,1]
        std::vector<double>
        get_quadrature_weights(const unsigned int n)
        {
            std::vector<double> weights(n);
            switch(n)
            {
                case 1:
                    weights[0] = 1.;
                    break;
                case 2:
                    weights[0] = 0.25;
                    weights[1] = 0.75;
                    break;
                case 3:
                    weights[0] = 1./9.;
                    weights[1] = (16.+std::sqrt(6))/36.;
                    weights[2] = (16.-std::sqrt(6))/36.;
                    break;
                default:
                    Assert(false, dealii::StandardExceptions::ExcNotImplemented());
                    break;
            }
            return weights;
        }
    }

    template<>
    QGaussRadau<1>::QGaussRadau(const unsigned int n, EndPoint ep)
        : dealii::Quadrature<1>(n),
        ep(ep)
    {
        Assert(n > 0, dealii::ExcMessage("Need at least one point for quadrature rules"));
        std::vector<double> p = 
          internal::QGaussRadau::get_quadrature_points(n,ep);
        std::vector<double> w = 
          internal::QGaussRadau::get_quadrature_weights(n);

        for ( unsigned int i = 0 ; i < this->size() ; ++i)
        {
            this->quadrature_points[i] = dealii::Point<1>(p[i]);
            this->weights[i]           = w[i];
        }
    }

    template<int dim>
    QGaussRadau<dim>::QGaussRadau(const unsigned int n,
                                  EndPoint           ep)
        : dealii::Quadrature<dim>(QGaussRadau<1>(
            n,
            static_cast<QGaussRadau<1>::EndPoint>(ep)))
        , ep(ep)
    {}
    

    QRightBox::QRightBox ()
        :
        QGaussRadau<1> ( 1 , QGaussRadau<1>::EndPoint::right )
    {}

    QLeftBox::QLeftBox ()
        :
        dealii::QGaussRadau<1> ( 1, QGaussRadau<1>::EndPoint::right )
    {}
    
    namespace spacetime
    {
        template<int dim>
        QGauss<dim>::QGauss ( unsigned int n_spatial ,
                              unsigned int n_temporal )
            :
            Quadrature<dim> ( std::make_shared<dealii::QGauss<dim>> ( n_spatial ) ,
                              std::make_shared<dealii::QGauss<1>> ( n_temporal )
            )
        {
        }

        template<int dim>
        QGaussRightBox<dim>::QGaussRightBox ( unsigned int n_spatial )
            :
            Quadrature<dim> ( std::make_shared<dealii::QGauss<dim>> ( n_spatial ) ,
                              std::make_shared<QRightBox> ()
            )
        {
        }

        template<int dim>
        QGaussLeftBox<dim>::QGaussLeftBox ( unsigned int n_spatial )
            :
            Quadrature<dim> ( std::make_shared<dealii::QGauss<dim>> ( n_spatial ) ,
                              std::make_shared<QLeftBox> ()
            )
        {
        }
    }
}

#include "quadrature_lib.inst"

