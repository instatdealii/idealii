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


#ifndef INCLUDE_IDEAL_II_LAC_SPACETIME_VECTOR_HH_
#define INCLUDE_IDEAL_II_LAC_SPACETIME_VECTOR_HH_

#include <deal.II/lac/vector.h>

namespace idealii{
namespace slab{
	/**
	 *	@brief A shortened type for iterators over a list of shared pointers to dealii::Vectors.
	 */
	template<typename Number>
	using VectorIterator = typename std::list<dealii::Vector<Number>>::iterator;
}
namespace spacetime{


	/**
	 * @brief The spacetime vector object.
	 *
	 * In practice this is just a class around a list of shared pointers to
	 * dealii::Vector<Number> objects to simplify time marching.
	 */
	template <typename Number>
	class Vector{
	public:
		/**
		 * @brief Construct an empty list of vectors.
		 */
		Vector();

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
		slab::VectorIterator<Number> begin();
		/**
		 * @brief Return an iterator pointing behind the last "slab" vector.
		 */
		slab::VectorIterator<Number> end();

	private:
		std::list<dealii::Vector<Number>> _vectors;
	};
}}



#endif /* INCLUDE_IDEAL_II_LAC_SPACETIME_VECTOR_HH_ */
