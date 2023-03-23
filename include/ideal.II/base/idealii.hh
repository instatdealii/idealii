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
#ifndef INCLUDE_IDEAL_II_BASE_IDEALII_HH_
#define INCLUDE_IDEAL_II_BASE_IDEALII_HH_
/**
 * @brief The main namespace of the project.
 */
namespace idealii
{
    void
    print_version_info ();

    /**
     * @brief Namespace for slab objects.
     *
     * Namespace containing all classes that describe objects
     * operating on a tensor product between a onedimensional
     * temporal triangulation and an n-dimensional spatial triangulation.
     */
    namespace slab
    {
        /**
         * @brief Namespace for MPI parallel objects in space.
         */
        namespace parallel
        {
            /**
             * @brief Namespace where the processor local triangulations share a common coarse object.
             */
            namespace distributed
            {
            }
        }
        /**
         * @brief Collection of functions working on degrees of freedom.
         *
         * These functions provide utilities for calculating
         * spacetime variants of normally stationary objects provided by deal.II.
         *
         * In detail this means spreading stationary information over all temporal degrees of freedom
         * in this slab.
         * Examples are:
         * - Offsetting hanging node constraints to each temporal dof.
         * - Building tensor products of spatial and temporal sparsity patterns to obtain a spacetime pattern
         */
        namespace DoFTools
        {
        }

        /**
         * @brief Collection of functions working on space-time slab Vectors.
         *
         * These functions provide utilities for manipulating space-time indexed
         * vectors of normally stationary objects provided by deal.II.
         *
         * Examples are:
         * - interpolation of spatial boundary values to all corresponding space-time dofs
         * - evaluation of space-time vectors at a specific time points of time dof.
         */
        namespace VectorTools
        {
        }
    }

    /**
     * @brief Namespace for general spacetime object and collections of slab objects.
     *
     * Namespace containing two types of classes.
     *
     * The first are general spacetime definitions independent of the actual triangulation.
     * Examples are space-time finite element classes and quadrature formulae.
     *
     * The second are classes containing lists of slab:: classes or general objects
     * related to a specific slab.
     * Examples are triangulations, DoF handlers and vectors.
     */
    namespace spacetime
    {
        /**
         * @brief Namespace for tensor product triangulations with a single fixed spatial mesh.
         */
        namespace fixed
        {
        }
        /**
         * @brief Namespace for MPI parallel objects in space.
         */
        namespace parallel
        {
            /**
             * @brief Namespace where the processor local triangulations share a common coarse object.
             */
            namespace distributed
            {
                /**
                 * @brief Namespace for tensor product triangulations with a single fixed spatial mesh.
                 */
                namespace fixed
                {
                }
            }
        }
    }
}
#endif /* INCLUDE_IDEAL_II_BASE_IDEALII1_HH_ */
