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

#include <ideal.II/dofs/slab_dof_handler.hh>

namespace idealii::slab
{
    template<int dim>
    DoFHandler<dim>::DoFHandler ( Triangulation<dim> &tria )
    {
        Assert( tria.spatial ().use_count () ,
                dealii::ExcNotInitialized () );
        Assert( tria.temporal ().use_count () ,
                dealii::ExcNotInitialized () );
        _spatial_dof = std::make_shared<dealii::DoFHandler<dim>> (
                *tria.spatial () );
        _temporal_dof = std::make_shared<dealii::DoFHandler<1>> (
                *tria.temporal () );
        _locally_owned_dofs = dealii::IndexSet ();
    }

    template<int dim>
    DoFHandler<dim>::DoFHandler (
            slab::parallel::distributed::Triangulation<dim> &tria )
    {
        Assert( tria.spatial ().use_count () ,
                dealii::ExcNotInitialized () );
        Assert( tria.temporal ().use_count () ,
                dealii::ExcNotInitialized () );
        _spatial_dof = std::make_shared<dealii::DoFHandler<dim>> (
                *tria.spatial () );
        _temporal_dof = std::make_shared<dealii::DoFHandler<1>> (
                *tria.temporal () );
        _locally_owned_dofs = dealii::IndexSet ();
    }

    template<int dim>
    DoFHandler<dim>::DoFHandler ( const DoFHandler<dim> &other )
    {
        Assert( other._spatial_dof.use_count () ,
                dealii::ExcNotInitialized () );
        _spatial_dof = other._spatial_dof;
        Assert( other._temporal_dof.use_count () ,
                dealii::ExcNotInitialized () );
        _temporal_dof = other._temporal_dof;
        _locally_owned_dofs = other._locally_owned_dofs;
        _fe_support_type = other._fe_support_type;
    }

    template<int dim>
    std::shared_ptr<dealii::DoFHandler<dim>> DoFHandler<dim>::spatial ()
    {
        return _spatial_dof;
    }

    template<int dim>
    std::shared_ptr<dealii::DoFHandler<1>> DoFHandler<dim>::temporal ()
    {
        return _temporal_dof;
    }

    template<int dim>
    void DoFHandler<dim>::distribute_dofs (
            spacetime::DG_FiniteElement<dim> fe )
    {
        _fe_support_type = fe.type ();
        _spatial_dof->distribute_dofs ( *fe.spatial () );
        _temporal_dof->distribute_dofs ( *fe.temporal () );
        dealii::IndexSet space_owned_dofs =
                _spatial_dof->locally_owned_dofs ();

        _locally_owned_dofs.clear ();
        _locally_owned_dofs.set_size (
                space_owned_dofs.size () * _temporal_dof->n_dofs () );

        for ( dealii::types::global_dof_index time_dof_index
                { 0 } ; time_dof_index < _temporal_dof->n_dofs () ;
                time_dof_index++ )
        {
            _locally_owned_dofs.add_indices (
                    space_owned_dofs ,
                    time_dof_index * _spatial_dof->n_dofs () // offset
            );
        }
    }

    template<int dim>
    unsigned int DoFHandler<dim>::n_dofs_spacetime ()
    {
        return _spatial_dof->n_dofs () * _temporal_dof->n_dofs ();
    }

    template<int dim>
    unsigned int DoFHandler<dim>::n_dofs_space ()
    {
        return _spatial_dof->n_dofs ();
    }

    template<int dim>
    unsigned int DoFHandler<dim>::n_dofs_time ()
    {
        return _temporal_dof->n_dofs ();
    }

    template<int dim>
    unsigned int DoFHandler<dim>::dofs_per_cell_time ()
    {
        return _temporal_dof->get_fe ().dofs_per_cell;
    }

    template<int dim>
    typename spacetime::DG_FiniteElement<dim>::support_type DoFHandler<dim>::fe_support_type ()
    {
        return _fe_support_type;
    }

    template<int dim>
    const dealii::IndexSet& DoFHandler<dim>::locally_owned_dofs ()
    {
        return _locally_owned_dofs;
    }
}
#include "slab_dof_handler.inst"

