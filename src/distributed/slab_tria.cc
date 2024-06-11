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

#include <deal.II/base/config.h>

#include <ideal.II/distributed/slab_tria.hh>

#ifdef DEAL_II_WITH_MPI
#  include <deal.II/grid/grid_generator.h>

namespace idealii::slab::parallel::distributed
{

  template <int dim>
  Triangulation<dim>::Triangulation(
    std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>>
           space_tria,
    double start,
    double end)
    : _startpoint(start)
    , _endpoint(end)
  {
    Assert(space_tria.use_count(), dealii::ExcNotInitialized());
    _spatial_tria  = space_tria;
    _temporal_tria = std::make_shared<dealii::Triangulation<1>>();
    dealii::GridGenerator::hyper_cube(*_temporal_tria, _startpoint, _endpoint);
  }


  template <int dim>
  Triangulation<dim>::Triangulation(
    std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>>
                        space_tria,
    std::vector<double> step_sizes,
    double              start,
    double              end)
    : _startpoint(start)
    , _endpoint(end)
  {
    _spatial_tria  = space_tria;
    _temporal_tria = std::make_shared<dealii::Triangulation<1>>();
    // Grid generator needs step sizes for each dimension,
    // so we need to construct a new vector with on entry.
    std::vector<std::vector<double>> spacing;
    spacing.push_back(step_sizes);
    dealii::Point<1> p1(_startpoint);
    dealii::Point<1> p2(_endpoint);
    dealii::GridGenerator::subdivided_hyper_rectangle(*_temporal_tria,
                                                      spacing,
                                                      p1,
                                                      p2);
  }

  template <int dim>
  Triangulation<dim>::Triangulation(const Triangulation &other)
    : _startpoint(other._startpoint)
    , _endpoint(other._endpoint)
  {
    Assert(other._spatial_tria.use_count(), dealii::ExcNotInitialized());
    _spatial_tria = other._spatial_tria;
    Assert(other._temporal_tria.use_count(), dealii::ExcNotInitialized());
    _temporal_tria = other._temporal_tria;
  }

  template <int dim>
  std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>>
  Triangulation<dim>::spatial()
  {
    Assert(_spatial_tria.use_count(), dealii::ExcNotInitialized());
    return _spatial_tria;
  }

  template <int dim>
  std::shared_ptr<dealii::Triangulation<1>>
  Triangulation<dim>::temporal()
  {
    Assert(_temporal_tria.use_count(), dealii::ExcNotInitialized());
    return _temporal_tria;
  }

  template <int dim>
  double
  Triangulation<dim>::startpoint()
  {
    return _startpoint;
  }

  template <int dim>
  double
  Triangulation<dim>::endpoint()
  {
    return _endpoint;
  }

  template <int dim>
  void
  Triangulation<dim>::update_temporal_triangulation(
    std::vector<double> step_sizes,
    double              startpoint,
    double              endpoint)
  {
    _startpoint = startpoint;
    _endpoint   = endpoint;
    std::vector<std::vector<double>> spacing;
    spacing.push_back(step_sizes);
    // TODO: add an assertion that no subscribers exist
    _temporal_tria->clear();
    dealii::Point<1> p1(_startpoint);
    dealii::Point<1> p2(_endpoint);
    dealii::GridGenerator::subdivided_hyper_rectangle(*_temporal_tria,
                                                      spacing,
                                                      p1,
                                                      p2);
  }
} // namespace idealii::slab::parallel::distributed

#  include "slab_tria.inst"
#endif
