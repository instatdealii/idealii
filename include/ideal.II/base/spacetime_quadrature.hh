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

#ifndef INCLUDE_IDEAL_II_BASE_SPACETIME_QUADRATURE_HH_
#define INCLUDE_IDEAL_II_BASE_SPACETIME_QUADRATURE_HH_

#include <deal.II/base/quadrature.h>

namespace idealii{
namespace spacetime{
	/**
	 * @brief The base class for quadrature formulae in space and time.
	 *
	 * This is simply a convenience class that holds shared pointers to a spatial
	 * and a temporal quadrature formula.
	 *
	 * Therefore, it can be used with any user supplied formula.
	 * For common combinations like Gauss-Legendre in space and time derived classes
	 * may exist to simplify construction.
	 */
	template<int dim>
	class Quadrature{
	public:
		/**
		 * @brief Construct a spacetime quadrature formula by supplying shared pointers to
		 * a spatial and a temporal quadrature formula.
		 */
		Quadrature(std::shared_ptr<dealii::Quadrature<dim>> quad_space,
				   std::shared_ptr<dealii::Quadrature<1>> quad_time);
		/**
		 * @brief The underlying spatial quadrature formula
		 * @return A shared pointer to the spatial quadrature formula.
		 */
		std::shared_ptr<dealii::Quadrature<dim>> spatial();
		/**
		 * @brief The underlying temporal quadrature formula
		 * @return A shared pointer to the temporal quadrature formula.
		 */
		std::shared_ptr<dealii::Quadrature<1>> temporal();
	private:
		std::shared_ptr<dealii::Quadrature<dim>> _quad_space;
		std::shared_ptr<dealii::Quadrature<1>> _quad_time;
	};
}}


#endif /* INCLUDE_IDEAL_II_BASE_SPACETIME_QUADRATURE_HH_ */
