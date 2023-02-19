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


#ifndef INCLUDE_IDEAL_II_FE_FE_DG_HH_
#define INCLUDE_IDEAL_II_FE_FE_DG_HH_


#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>

#include <memory>

namespace idealii{
  namespace spacetime{
    template<int dim>
    /**
     * @brief A class for dG elements in time and arbitrary elements in space.
     *
     * This class implements a tensor product finite element of a discontinuous
     * Galerkin temporal element and a user supplied spatial element.
     * For the temporal element both GaussLegendre and GaussLobatto support points
     * can be chosen
     **/
    class DG_FiniteElement
    {
    public:
    	//Info/Todo: With proper Quadrature class could be extended to RaudauLeft and RadauRight
    	//Would need Radau based on Legendre polynomial roots
    	/**
    	 * @brief Choice of underlying temporal support points.
    	 *
    	 * Allows the choice between GaussLobatto and GaussLegendre support points
    	 * for the dG(r) elements in time.
    	 *
    	 * The choice of #Lobatto results in 1D dG elements as defined in FE_DGQ.
    	 *
    	 * The choice of #Legendre leads to dG elements without support points on the
    	 * interval/element edges. This might lead to better convergence, but the
    	 * resulting off-diagonal sparsity pattern will be larger as jump terms
    	 * then depend on all temporal dofs.
    	 *
    	 * For r=0 the resulting element always uses the interval midpoint.
    	 */
    	enum support_type{
    		/**Support points based on QGauss<1>*/
    		Legendre,
			/**for dG(r), r>0: Support points based on QGaussLobatto<1>*/
			Lobatto
    	};
    	/**
    	 * @brief Constructor for the finite element class.
    	 */
    	DG_FiniteElement(std::shared_ptr<dealii::FiniteElement<dim>> fe_space,
    					 const unsigned int r,
						 support_type type = support_type::Lobatto);
    	/**
    	 * @brief The underlying spatial finite element.
    	 * @return A shared pointer to the underlying finite element object.
    	 */
    	std::shared_ptr<dealii::FiniteElement<dim>> spatial();
    	/**
    	 * @brief The underlying temporal finite element.
    	 * @return A shared pointer to the underlying finite element object.
    	 */
    	std::shared_ptr<dealii::FiniteElement<1>> temporal();
    	/**
    	 * @brief The number of degrees of freedom per space-time element.
    	 *
    	 * In practice this is simply (r+1) times the number of DoFs per spatial element.
    	 * @return temporal()->dofs_per_cell * spatial()->dofs_per_cell
    	 */
    	const unsigned int dofs_per_cell;
    	/**
    	 * @brief The #support_type used for construction.
    	 *
    	 * This function is mostly used internally. One example would be the construction
    	 * of the upstream and downstream sparsity patterns,
    	 * which include off diagonals for jump terms.
    	 * @return The underlying #support_type
    	 */
    	support_type type();
    private:
    	std::shared_ptr<dealii::FiniteElement<dim>> _fe_space;
    	std::shared_ptr<dealii::FiniteElement<1>> _fe_time;
    	support_type _type;

    };
  } //end namespace idealii
} //end namespace idealii



#endif /* INCLUDE_IDEAL_II_FE_FE_DG_HH_ */
