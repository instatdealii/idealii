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
#include <deal.II/base/quadrature_lib.h>
namespace idealii{
	QRightBox::QRightBox():
		dealii::Quadrature<1>(1){
		this->quadrature_points[0] = dealii::Point<1>(1.0);
		this->weights[0] = 1.0;
	}

	QLeftBox::QLeftBox():
		dealii::Quadrature<1>(1){
		this->quadrature_points[0] = dealii::Point<1>(0.0);
		this->weights[0] = 1.0;
	}

namespace spacetime{
	template <int dim>
	QGauss<dim>::QGauss(unsigned int n_spatial, unsigned int n_temporal):
		Quadrature<dim>(std::make_shared<dealii::QGauss<dim>>(n_spatial),
									   std::make_shared<dealii::QGauss<1>>(n_temporal))
	{}

	template <int dim>
	QGaussRightBox<dim>::QGaussRightBox(unsigned int n_spatial):
		Quadrature<dim>(std::make_shared<dealii::QGauss<dim>>(n_spatial),
		   								    std::make_shared<QRightBox>())
	{}

	template <int dim>
	QGaussLeftBox<dim>::QGaussLeftBox(unsigned int n_spatial):
		Quadrature<dim>(std::make_shared<dealii::QGauss<dim>>(n_spatial),
		   								    std::make_shared<QLeftBox>())
	{}
}}


#include "quadrature_lib.inst"

