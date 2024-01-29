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

#ifndef INCLUDE_IDEAL_II_DOFS_SLAB_DOF_HANDLER_HH_
#define INCLUDE_IDEAL_II_DOFS_SLAB_DOF_HANDLER_HH_

#include <ideal.II/grid/slab_tria.hh>
#include <ideal.II/distributed/slab_tria.hh>

#include <ideal.II/fe/fe_dg.hh>

#include <deal.II/dofs/dof_handler.h>

#include <memory>
#include <list>

namespace idealii::slab
{
    /**
     * @brief Actual DoFHandler for a specific slab.
     *
     * This DoFHandler actually handles a spatial and a temporal
     * dealii::DoFHandler object internally.
     *
     * Currently it is restricted to dG elements in time i.e. spacetime::DG_FiniteElement.
     */
    template<int dim>
    class DoFHandler
    {
    public:
        /**
         * @brief Constructor linking a slab::Triangulation.
         * @param tria A shared pointer to a slab::Triangulation.
         */
        DoFHandler ( Triangulation<dim> &tria );
#ifdef DEAL_II_WITH_MPI
        /**
         * @brief Constructor linking a parallel::distributed::slab::Triangulation.
         * @param tria A shared pointer to a parallel::distributed::slab::Triangulation.
         */
        DoFHandler (
                slab::parallel::distributed::Triangulation<dim> &tria );
#endif
        /**
         * @brief (shallow) copy constructor. Only the index set and fe support type
         * are actually copied. The underlying pointers will point to the same
         * dealii::DoFHandler objects as other.
         *
         * @param other The DoFHandler to shallow copy.
         *
         * @warning This method is mainly needed for adding the DoFHandlers to the spacetime
         * lists and should be used with utmost caution anywhere else.
         *
         */

        DoFHandler ( const DoFHandler<dim> &other );
        /**
         * @brief The underlying spatial dof handler.
         * @return A shared pointer to the spatial dof handler.
         */
        std::shared_ptr<dealii::DoFHandler<dim>>
        spatial ();

        /**
         * @brief The underlying temporal dof handler.
         * @return A shared pointer to the temporal dof handler.
         */
        std::shared_ptr<dealii::DoFHandler<1>>
        temporal ();

        /**
         * @brief Distribute DoFs in space and time.
         *
         * This function calls the function of the same name of the underlying
         * dof handler objects with the matching spatial or temporal element in @p fe.
         *
         * @param fe The spacetime finite element to base the distribution on.
         */
        void
        distribute_dofs ( spacetime::DG_FiniteElement<dim> fe );

        /**
         * @brief Total number of space-time degrees of fredom on this slab.
         *
         * @return The total number of space-time dofs, i.e. n_dofs_space()*n_dofs_time().
         */
        unsigned int
        n_dofs_spacetime ();
        /**
         * @brief Number of spatial degrees of fredom on this slab.
         *
         * @return The number of dofs based on the spatial finite element and triangulation.
         */
        unsigned int
        n_dofs_space ();
        /**
         * @brief Number of temporal degrees of fredom on this slab.
         *
         * @return The number of dofs based on the temporal finite element and  triangulation.
         */
        unsigned int
        n_dofs_time ();

        /**
         * @brief Number of temporal dofs in a single element/interval.
         *
         * @return The number of temporal dofs i.e. (r+1) for dG(r) elements.
         */
        unsigned int
        dofs_per_cell_time ();

        /**
         * @brief The underlying support type used for constructing the temporal finite element.
         *
         * @return The spacetime::DG_FiniteElement<dim>::support_type of the underlying finite element.
         */
        typename spacetime::DG_FiniteElement<dim>::support_type
        fe_support_type ();

        /**
         * @brief Return the set of processor local dofs.
         *
         * Parallelization is done in space only. Therefore the local space-time dof indices
         * are the local indices of the spatial DoFHandler shifted by the total number of
         * spatial degrees of freedom.
         * Note that the IndexSet is not contiguous and can therefore currently not be used
         * with PETScWrapper classes.
         *
         * @return An IndexSet of the global dof indices owned by the current core.
         */
        const dealii::IndexSet&
        locally_owned_dofs ();

    private:
        std::shared_ptr<dealii::DoFHandler<dim>> _spatial_dof;
        std::shared_ptr<dealii::DoFHandler<1>> _temporal_dof;
        typename spacetime::DG_FiniteElement<dim>::support_type _fe_support_type;
        dealii::IndexSet _locally_owned_dofs;
    };

    /**
     * @brief A shortened type for iterators over a list of shared pointers to DoFHandler objects.
     */
    template<int dim>
    using DoFHandlerIterator = typename std::list<DoFHandler<dim>>::iterator;
}

#endif /* INCLUDE_IDEAL_II_DOFS_SLAB_DOF_HANDLER_HH_ */
