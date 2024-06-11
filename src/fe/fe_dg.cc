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

#include "ideal.II/base/quadrature_lib.hh"

#include <ideal.II/fe/fe_dg.hh>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>

#include <memory>

namespace idealii::spacetime
{
  template <int dim>
  DG_FiniteElement<dim>::DG_FiniteElement(
    std::shared_ptr<dealii::FiniteElement<dim>> fe_space,
    const unsigned int                          r,
    support_type                                type)
    : dofs_per_cell(fe_space->dofs_per_cell * (r + 1))
    , _fe_space(fe_space)
    , _type(type)
  {
    if (type == support_type::Legendre ||
        (type == support_type::Lobatto && r == 0))
      {
        _fe_time = std::make_shared<dealii::FE_DGQArbitraryNodes<1>>(
          dealii::QGauss<1>(r + 1));
      }
    else if (type == support_type::Lobatto)
      {
        _fe_time = std::make_shared<dealii::FE_DGQArbitraryNodes<1>>(
          dealii::QGaussLobatto<1>(r + 1));
      }
    else if (type == support_type::RadauLeft)
      {
        _fe_time = std::make_shared<dealii::FE_DGQArbitraryNodes<1>>(
          idealii::QGaussRadau<1>(r + 1, QGaussRadau<1>::left));
      }
    else
      {
        _fe_time = std::make_shared<dealii::FE_DGQArbitraryNodes<1>>(
          idealii::QGaussRadau<1>(r + 1, QGaussRadau<1>::right));
      }
  }

  template <int dim>
  std::shared_ptr<dealii::FiniteElement<dim>>
  DG_FiniteElement<dim>::spatial()
  {
    return _fe_space;
  }

  template <int dim>
  std::shared_ptr<dealii::FiniteElement<1>>
  DG_FiniteElement<dim>::temporal()
  {
    return _fe_time;
  }

  template <int dim>
  typename DG_FiniteElement<dim>::support_type
  DG_FiniteElement<dim>::type()
  {
    return _type;
  }
} // namespace idealii::spacetime

#include "fe_dg.inst"
