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

#ifndef INCLUDE_IDEAL_II_DOFS_SPACETIME_DOF_HANDLER_HH_
#define INCLUDE_IDEAL_II_DOFS_SPACETIME_DOF_HANDLER_HH_

#include <ideal.II/grid/spacetime_tria.hh>
#include <ideal.II/distributed/spacetime_tria.hh>
#include <ideal.II/dofs/slab_dof_handler.hh>

namespace idealii::spacetime
{
    /**
     * @brief The spacetime dofhandler object.
     *
     * In practice this is just a class around a list of shared pointers to
     * slab::DoFHandler objects to simplify generation and time marching.
     */
    template<int dim>
    class DoFHandler
    {
    public:
        /**
         * @brief Constructor based on spacetime::Triangulation.
         *
         * @param tria The spacetime::Triangulation object to use in construction of the underlying handlers.
         */
        DoFHandler ( spacetime::Triangulation<dim> *tria );

#ifdef DEAL_II_WITH_MPI
        /**
         * @brief Constructor based on parallel::distributed::spacetime::Triangulation.
         *
         * @param tria The parallel::distributed::spacetime::Triangulation object to use in construction of the underlying handlers.
         */
        DoFHandler(spacetime::parallel::distributed::Triangulation<dim>* tria);
#endif
        /**
         * @brief generate all slab::DofHandler objects.
         *
         * This function iterates over all slab::Triangulation objects in the
         * underlying triangulation and constructs one slab::DoFHandler for each.
         */
        void
        generate ();

        /**
         * @brief The number of slabs.
         * @return The size of the underlying list.
         */
        unsigned int
        M ();

        /**
         * @brief An iterator pointing to the first slab::DoFHandler.
         * @return The result of the begin() call to the underlying list.
         */
        slab::DoFHandlerIterator<dim>
        begin ();
        /**
         * @brief An iterator pointing behind the last slab::DoFHandler.
         * @return The result of the end() call to the underlying list.
         */
        slab::DoFHandlerIterator<dim>
        end ();

    protected:
        Triangulation<dim> *_tria;
        spacetime::parallel::distributed::Triangulation<dim> *_par_dist_tria;
        std::list<slab::DoFHandler<dim>> _dof_handlers;
    };
}

#endif /* INCLUDE_IDEAL_II_DOFS_FIXED_DOF_HANDLER_HH_ */
