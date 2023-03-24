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

#ifndef INCLUDE_IDEAL_II_DISTRIBUTED_SLAB_TRIA_HH_
#define INCLUDE_IDEAL_II_DISTRIBUTED_SLAB_TRIA_HH_

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_MPI
#include <deal.II/distributed/tria.h>
#include <memory>
#include <list>

namespace idealii::slab::parallel::distributed{
    /**
     * @brief Actual Triangulation for a specific slab with an MPI distributed spatial mesh.
     *
     * This Triangulation handles a spatial dealii::parallel::distributed and
     * a temporal dealii::Triangulation object internally.
     *
     */
    template<int dim>
    class Triangulation{
    public:

        /**
         * @brief Construct an object with a given spatial triangulation and a single
         * element in time.
         *
         * @param space_tria The spatial triangulation to be used.
         * @param startpoint The startpoint of the temporal triangulation.
         * @param endpoint The endpoint of the temporal triangulation.
         */
        Triangulation(std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> space_tria,
                      double startpoint,
                      double endpoint);

        /**
         * @brief (shallow) copy constructor. Only the values for the start- and endpoint
         * are actually copied. The underlying pointers will point to the same
         * dealii::Triangulation objects as other.
         *
         * @param other The Triangulation to shallow copy.
         *
         * @warning This method is mainly needed for adding the Triangulations to the spacetime
         * lists and should be used with utmost caution anywhere else.
         *
         *
         */
        Triangulation(const Triangulation& other);
        /**
         * @brief The underlying spatial triangulation.
         */
        std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> spatial();

        /**
         * @brief The underlying temporal triangulation.
         */
        std::shared_ptr<dealii::Triangulation<1>> temporal();

        /**
         * @brief The startpoint of the temporal triangulation.
         */
        double startpoint();

        /**
         * @brief The endpoint of the temporal triangulation.
         */
        double endpoint();

    private:
        std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> _spatial_tria;
        std::shared_ptr<dealii::Triangulation<1>> _temporal_tria;
        double _startpoint;
        double _endpoint;
    };

    /**
     * @brief A shortened type for Iterators over a list of shared pointers to Triangulation<dim> objects
     */
    template<int dim>
    using TriaIterator = typename std::list<Triangulation<dim>>::iterator;
}
#endif
#endif /* INCLUDE_IDEAL_II_DISTRIBUTED_SLAB_TRIA_HH_ */
