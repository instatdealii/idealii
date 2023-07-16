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

#ifndef INCLUDE_IDEAL_II_GRID_FIXED_TRIA_HH_
#define INCLUDE_IDEAL_II_GRID_FIXED_TRIA_HH_

#include <memory>
#include <list>
#include "spacetime_tria.hh"

namespace idealii::spacetime::fixed
{
    /**
     * @brief The spacetime triangulation object with a fixed spatial mesh across time.
     *
     * In practice all pointers in the list point to the same slab::Triangulation object.
     */
    template<int dim>
    class Triangulation : public spacetime::Triangulation<dim>
    {
    public:
        /**
         * @brief Constructor that initializes the underlying list object.
         * @param max_N_intervals_per_slab. When to split a slab into two. (default 0 = never)
         */
        Triangulation ( unsigned int max_N_intervals_per_slab=0);

        /**
         * @brief Generate a list of M slab triangulations with matching temporal meshes pointing to the same
         * spatial triangulation.
         * @param space_tria The underlying spatial dealii::Triangulation to be used by all slabs.
         * @param M The number of slabs to be created
         * @param t0 The temporal startpoint. Defaults to 0.
         * @param T The temporal endpoint. Defaults to 1.
         */
        void generate (
                std::shared_ptr<dealii::Triangulation<dim>> space_tria ,
                unsigned int M , double t0 = 0. , double T = 1. );

        void refine_global ( const unsigned int times_space = 1 ,
                             const unsigned int times_time = 1 );
    };
}

#endif /* INCLUDE_IDEAL_II_GRID_FIXED_TRIA_HH_ */
