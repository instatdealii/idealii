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

#include <ideal.II/grid/slab_tria.hh>

#include <deal.II/grid/grid_generator.h>

namespace idealii::slab
{

    template<int dim>
    Triangulation<dim>::Triangulation (
            std::shared_ptr<dealii::Triangulation<dim>> space_tria ,
            double start , double end )
    :
            _startpoint ( start ), _endpoint ( end )
    {
        Assert( space_tria.use_count () , dealii::ExcNotInitialized () );
        _spatial_tria = space_tria;
        _temporal_tria = std::make_shared<dealii::Triangulation<1>> ();
        dealii::GridGenerator::hyper_cube ( *_temporal_tria , _startpoint ,
                                            _endpoint );
    }

    template<int dim>
    Triangulation<dim>::Triangulation ( const Triangulation &other )
    :
    _startpoint ( other._startpoint ), _endpoint ( other._endpoint )
    {
        Assert( other._spatial_tria.use_count () ,
                dealii::ExcNotInitialized () );
        _spatial_tria = other._spatial_tria;
        Assert( other._temporal_tria.use_count () ,
                dealii::ExcNotInitialized () );
        _temporal_tria = other._temporal_tria;
    }

    template<int dim>
    std::shared_ptr<dealii::Triangulation<dim>> Triangulation<dim>::spatial ()
    {
        Assert( _spatial_tria.use_count () , dealii::ExcNotInitialized () );
        return _spatial_tria;
    }

    template<int dim>
    std::shared_ptr<dealii::Triangulation<1>> Triangulation<dim>::temporal ()
    {
        Assert( _temporal_tria.use_count () , dealii::ExcNotInitialized () );
        return _temporal_tria;
    }

    template<int dim>
    double Triangulation<dim>::startpoint ()
    {
        return _startpoint;
    }

    template<int dim>
    double Triangulation<dim>::endpoint ()
    {
        return _endpoint;
    }
}

#include "slab_tria.inst"

