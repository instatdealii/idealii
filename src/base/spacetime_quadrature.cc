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

#include <ideal.II/base/spacetime_quadrature.hh>

namespace idealii::spacetime
{

    template<int dim>
    Quadrature<dim>::Quadrature (
            std::shared_ptr<dealii::Quadrature<dim>> quad_space ,
            std::shared_ptr<dealii::Quadrature<1>> quad_time )
    {
        Assert( quad_space.use_count () , dealii::ExcNotImplemented () );
        Assert( quad_time.use_count () , dealii::ExcNotImplemented () );
        _quad_space = quad_space;
        _quad_time = quad_time;
    }

    template<int dim>
    std::shared_ptr<dealii::Quadrature<dim>> Quadrature<dim>::spatial ()
    {
        return _quad_space;
    }

    template<int dim>
    std::shared_ptr<dealii::Quadrature<1>> Quadrature<dim>::temporal ()
    {
        return _quad_time;
    }
}

#include "spacetime_quadrature.inst"

