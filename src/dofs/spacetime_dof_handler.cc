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

#include <ideal.II/dofs/spacetime_dof_handler.hh>

namespace idealii::spacetime
{
  template <int dim>
  DoFHandler<dim>::DoFHandler(Triangulation<dim> *tria)
  {
    _tria = tria;
#ifdef DEAL_II_WITH_MPI
    _par_dist_tria = nullptr;
#endif
    _dof_handlers = std::list<idealii::slab::DoFHandler<dim>>();
  }

#ifdef DEAL_II_WITH_MPI
  template <int dim>
  DoFHandler<dim>::DoFHandler(
    spacetime::parallel::distributed::Triangulation<dim> *tria)
  {
    _tria          = nullptr;
    _par_dist_tria = tria;
    _dof_handlers  = std::list<idealii::slab::DoFHandler<dim>>();
  }
#endif
  template <int dim>
  unsigned int
  DoFHandler<dim>::M()
  {
    return _dof_handlers.size();
  }

  template <int dim>
  void
  DoFHandler<dim>::generate()
  {
    if (_tria != nullptr)
      {
        slab::TriaIterator<dim> tria_it  = this->_tria->begin();
        slab::TriaIterator<dim> tria_end = this->_tria->end();
        for (; tria_it != tria_end; ++tria_it)
          {
            this->_dof_handlers.push_back(
              idealii::slab::DoFHandler<dim>(*tria_it));
          }
      }
#ifdef DEAL_II_WITH_MPI
    else if (_par_dist_tria != nullptr)
      {
        slab::parallel::distributed::TriaIterator<dim> tria_it =
          this->_par_dist_tria->begin();
        slab::parallel::distributed::TriaIterator<dim> tria_end =
          this->_par_dist_tria->end();
        for (; tria_it != tria_end; ++tria_it)
          {
            this->_dof_handlers.push_back(
              idealii::slab::DoFHandler<dim>(*tria_it));
          }
      }
#endif
    else
      {
        Assert(false, dealii::ExcInternalError());
      }
  }

  template <int dim>
  slab::DoFHandlerIterator<dim>
  DoFHandler<dim>::begin()
  {
    return _dof_handlers.begin();
  }

  template <int dim>
  slab::DoFHandlerIterator<dim>
  DoFHandler<dim>::end()
  {
    return _dof_handlers.end();
  }
} // namespace idealii::spacetime

#include "spacetime_dof_handler.inst"
