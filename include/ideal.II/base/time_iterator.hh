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

#ifndef INCLUDE_IDEAL_II_BASE_TIME_ITERATOR_HH_
#define INCLUDE_IDEAL_II_BASE_TIME_ITERATOR_HH_

#include <ideal.II/distributed/spacetime_tria.hh>

#include <ideal.II/dofs/spacetime_dof_handler.hh>

#include <ideal.II/grid/spacetime_tria.hh>

#include <ideal.II/lac/spacetime_trilinos_vector.hh>
#include <ideal.II/lac/spacetime_vector.hh>

#include <list>
#include <memory>
#include <vector>
namespace idealii
{

  /**
   * @brief A collection of slab iterators to simplify time marching.
   *
   * The second type of spacetime objects like spacetime::Triangulation,
   * spacetime::DoFHandler and spacetime::Vector
   * are simply doubly linked lists of objects corresponding to a specific slab.
   *
   * For forward or backward time marching these lists need to be traversed
   * simultaneously. This class combines all iterator increments or decrements
   * into a single function call.
   */
  template <int dim>
  class TimeIteratorCollection
  {
  public:
    /**
     * @brief Default constructor.
     *
     * @note Any iterators have to be registered using one of the
     * add_iterator functions.
     */
    TimeIteratorCollection();
    /**
     * @brief Add a slab::TriaIterator<dim> iterator.
     *
     * @param it A pointer to the slab::TriaIterator<dim>
     * @param collection A pointer to the spacetime::Triangulation<dim> whose member list includes @p it
     *
     */
    void
    add_iterator(slab::TriaIterator<dim>       *it,
                 spacetime::Triangulation<dim> *collection);

#ifdef DEAL_II_WITH_MPI
    /**
     * @brief Add a parallel::distributed::slab::TriaIterator<dim> iterator.
     *
     * @param it A pointer to the parallel::distributed::slab::TriaIterator<dim>
     * @param collection A pointer to the parallel::distributed::spacetime::Triangulation<dim> whose member list includes @p it
     *
     */
    void
    add_iterator(
      slab::parallel::distributed::TriaIterator<dim>       *it,
      spacetime::parallel::distributed::Triangulation<dim> *collection);
#endif
    /**
     * @brief Add a slab::DoFHandlerIterator<dim> iterator.
     *
     * @param it A pointer to the slab::DoFHandlerIterator<dim>
     * @param collection A pointer to the spacetime::DoFHandler<dim> whose member list includes @p it
     *
     */
    void
    add_iterator(slab::DoFHandlerIterator<dim> *it,
                 spacetime::DoFHandler<dim>    *collection);

    /**
     * @brief Add a slab::VectorIterator<dim> iterator.
     *
     * @param it A pointer to the slab::VectorIterator<dim>
     * @param collection A pointer to the spacetime::Vector<dim> whose member list includes @p it
     *
     */
    void
    add_iterator(slab::VectorIterator<double> *it,
                 spacetime::Vector<double>    *collection);

#ifdef DEAL_II_WITH_TRILINOS
#  ifdef DEAL_II_WITH_MPI
    /**
     * @brief Add a slab::TrilinosVectorIterator<dim> iterator.
     *
     * @param it A pointer to the slab::TrilinosVectorIterator<dim>
     * @param collection A pointer to the spacetime::TrilinosVector<dim> whose member list includes @p it
     *
     */
    void
    add_iterator(slab::TrilinosVectorIterator *it,
                 spacetime::TrilinosVector    *collection);
#  endif
#endif
    /**
     * Increments all added iterators via their prefix increment operators.
     */
    void
    increment();
    /**
     * Decrements all added iterators via their prefix decrement operators.
     */
    void
    decrement();
    /**
     * For use as a stopping criterion in forward time marching.
     * @return true: if at least one of the iterators is a the corresponding end()
     * @return false: if none of the iterators is at the corresponding end()
     */
    bool
    at_end();
    /**
     * For use as a stopping criterion in backward time marching.
     * @return true: if at least one of the iterators is before the corresponding begin()
     * @return false: if none of the iterators is before the corresponding begin()
     */
    bool
    before_begin();

  private:
    struct
    {
      std::vector<slab::TriaIterator<dim> *>       it_collection;
      std::vector<spacetime::Triangulation<dim> *> obj_collection;
    } tria;

#ifdef DEAL_II_WITH_MPI
    struct
    {
      std::vector<slab::parallel::distributed::TriaIterator<dim> *>
        it_collection;
      std::vector<spacetime::parallel::distributed::Triangulation<dim> *>
        obj_collection;
    } par_dist_tria;
#endif
    struct
    {
      std::vector<slab::DoFHandlerIterator<dim> *> it_collection;
      std::vector<spacetime::DoFHandler<dim> *>    obj_collection;
    } dof;
    struct
    {
      std::vector<slab::VectorIterator<double> *> it_collection;
      std::vector<spacetime::Vector<double> *>    obj_collection;
    } vector_double;

#ifdef DEAL_II_WITH_TRILINOS
#  ifdef DEAL_II_WITH_MPI
    struct
    {
      std::vector<slab::TrilinosVectorIterator *> it_collection;
      std::vector<spacetime::TrilinosVector *>    obj_collection;
    } trilinos_vector;
#  endif
#endif
  };

} // namespace idealii

#endif /* INCLUDE_IDEAL_II_BASE_TIME_ITERATOR_HH_ */
