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


#ifndef INCLUDE_IDEAL_II_DISTRIBUTED_SPACETIME_TRIA_HH_
#define INCLUDE_IDEAL_II_DISTRIBUTED_SPACETIME_TRIA_HH_

#include <ideal.II/distributed/slab_tria.hh>

#ifdef DEAL_II_WITH_MPI
#include <deal.II/grid/tria.h>

#include <memory>
#include <list>

namespace idealii{
namespace spacetime{
namespace parallel{
namespace distributed{
	/**
	 * @brief The spacetime triangulation object with MPI parallel distributed spatial meshes.
	 *
	 * In practice this is just a class around a list of shared pointers to slab::parallel::distributed::Triangulation objects to
	 * simplify generation and time marching.
	 * @note This is a virtual base class.
	 */
  	  template<int dim>
  	  class Triangulation{
  	  public:
  		  /**
  		   * @brief Constructor that initializes the underlying list object.
  		   */
  		  Triangulation();

  		  /**
  		   * @brief Generate a list of M slab triangulations with matching temporal meshes and space_tria.
  		   *
  		   * @param space_tria The underlying spatial dealii::parallel::distributed::Triangulation.
  		   * @param M The number of slabs to be created.
  		   * @param t0 The temporal startpoint. Defaults to 0.
  		   * @param T The temporal endpoint. Defaults to 1.
  		   */
  		  virtual void generate(std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> space_tria,
  				  	  	  	    unsigned int M,
								double t0=0.,
								double T=1.)=0;

  		  /**
  		   * @brief Return the number of slabs in the triangulation.
  		   */
  		  unsigned int M();
  		  /**
  		   * brief An iterator pointing to the first slab::parallel::distributed::Triangulation
  		   */
  		  slab::parallel::distributed::TriaIterator<dim> begin();
  		  /**
  		   * @brief An iterator pointing behind the slab slab::parallel::distributed::Triangulation
  		   */
  		  slab::parallel::distributed::TriaIterator<dim> end();

  		  /**
  		   * @brief Do uniform mesh refinement in time and space
  		   * @param times_space Number of times the spatial meshes are refined.
  		   * @param times_time Number of times the temporal meshes are refined.
  		   */
  		  virtual void refine_global(const unsigned int times_space = 1, const unsigned int times_time = 1)=0;

  	  protected:
  	  	  std::list<slab::parallel::distributed::Triangulation<dim>> trias;
  	  };
}}}}


#endif
#endif /* INCLUDE_IDEAL_II_DISTRIBUTED_SPACETIME_TRIA_HH_ */
