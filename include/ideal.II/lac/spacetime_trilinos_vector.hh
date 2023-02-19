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


#ifndef INCLUDE_IDEAL_II_LAC_SPACETIME_TRILINOS_VECTOR_HH_
#define INCLUDE_IDEAL_II_LAC_SPACETIME_TRILINOS_VECTOR_HH_

#include <deal.II/base/config.h>
#ifdef DEAL_II_WITH_TRILINOS
#ifdef DEAL_II_WITH_MPI
#include <deal.II/lac/trilinos_vector.h>

namespace idealii{
namespace slab{
	/**
	 *	@brief A shortened type for iterators over a list of shared pointers to dealii::TrilinosWrappers::MPI::Vectors.
	 */
	using TrilinosVectorIterator = typename std::list<dealii::TrilinosWrappers::MPI::Vector>::iterator;
}
namespace spacetime{


	/**
	 * @brief The spacetime Trilinos vector object.
	 *
	 * In practice this is just a class around a list of shared pointers to
	 * dealii::TrilinosWrappers::MPI::Vector objects to simplify time marching.
	 */
	class TrilinosVector{
	public:
		/**
		 * @brief Construct an empty list of vectors.
		 */
		TrilinosVector();

		/**
		 * @brief Clear the list and add M empty vectors.
		 */
		void reinit(unsigned int M);

		/**
		 * @brief Return the size of the list, i.e. the number of slabs.
		 */
		unsigned int M();

		/**
		 * @brief Return an iterator pointing to the first "slab" vector.
		 */
		slab::TrilinosVectorIterator begin();
		/**
		 * @brief Return an iterator pointing behind the last "slab" vector.
		 */
		slab::TrilinosVectorIterator end();

	private:
		std::list<dealii::TrilinosWrappers::MPI::Vector> _vectors;
	};
}}

#endif /* DEAL_II_WITH_MPI */
#endif /* DEAL_II_WITH_TRILINOS */
#endif /* INCLUDE_IDEAL_II_LAC_SPACETIME_VECTOR_HH_ */
