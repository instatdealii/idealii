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

#include <ideal.II/distributed/fixed_tria.hh>

#ifdef DEAL_II_WITH_MPI
namespace idealii::spacetime::parallel::distributed::fixed
{
    template<int dim>
    Triangulation<dim>::Triangulation ()
    :
    spacetime::parallel::distributed::Triangulation<dim> ()
    {
    }

    template<int dim>
    void Triangulation<dim>::generate (
            std::shared_ptr<
            dealii::parallel::distributed::Triangulation<dim>> space_tria ,
            unsigned int M , double t0 , double T )
    {
        Assert( space_tria.use_count () , dealii::ExcNotInitialized () );
        //Todo: somehow assert that space_tria is actually parallel distributed
        double t = t0;
        double k = ( T - t0 ) / M;
        for ( unsigned int i = 0 ; i < M ; i++ )
        {
            this->trias.push_back (
                    idealii::slab::parallel::distributed::Triangulation<dim> (
                            space_tria , t , t + k ) );
            t += k;
        }
    }

    template<int dim>
    void Triangulation<dim>::refine_global (
            const unsigned int times_space , const unsigned int times_time )
    {

        slab::parallel::distributed::TriaIterator<dim> slab_tria =
                this->begin ();
        slab_tria->spatial ()->refine_global ( times_space );
        for ( ; slab_tria != this->end () ; ++slab_tria )
        {
            slab_tria->temporal ()->refine_global ( times_time );
        }

    }
}

#include "fixed_tria.inst"

#endif
