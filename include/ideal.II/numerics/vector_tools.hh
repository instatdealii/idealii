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

#ifndef INCLUDE_IDEAL_II_NUMERICS_VECTOR_TOOLS_HH_
#define INCLUDE_IDEAL_II_NUMERICS_VECTOR_TOOLS_HH_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>

#include <ideal.II/dofs/slab_dof_handler.hh>
#include <ideal.II/base/quadrature_lib.hh>

#include <memory>

namespace idealii::slab::VectorTools
{

    /**
     * @brief Compute space-time constraints on the solution corresponding to
     * Dirichlet conditions.
     * This function iterates over all temporal degrees of freedom and sets the
     * boundary values of the corresponding subvector to the value of the specified
     * at the time of the temporal dof.
     *
     * @param dof_handler The space-time slab dof handler the vector is indexed on
     * @param boundary_component The boundary id corresponding to the Dirichlet conditions
     * @param boundary_function The function that describes the Dirichlet data
     * @param spacetime_constraints The constraints to add the boundary values to
     * @param component_mask A component mask to only apply boundary values to certain FE components
     *
     * @note The set_time() method of boundary_function is called internally, so if the function is time-dependent use
     * get_time() in your implementation of the function to obtain t.
     */
    template<int dim,typename Number>
    void interpolate_boundary_values (
            idealii::slab::DoFHandler<dim> &dof_handler ,
            const dealii::types::boundary_id boundary_component ,
            dealii::Function<dim,Number> &boundary_function ,
            std::shared_ptr<dealii::AffineConstraints<Number>> spacetime_constraints ,
            const dealii::ComponentMask &component_mask =
                    dealii::ComponentMask () );

    /**
     * @brief Compute space-time constraints on the solution corresponding to
     * Dirichlet conditions for the locally relevant set of space dofs.
     * This function iterates over all temporal degrees of freedom and sets the
     * boundary values of the corresponding subvector to the value of the specified
     * at the time of the temporal dof.
     *
     * @param space_relevant_dofs The IndexSet of locally relevant dofs in space.
     * @param dof_handler The space-time slab dof handler the vector is indexed on
     * @param boundary_component The boundary id corresponding to the Dirichlet conditions
     * @param boundary_function The function that describes the Dirichlet data
     * @param spacetime_constraints The constraints to add the boundary values to
     * @param component_mask A component mask to only apply boundary values to certain FE components
     *
     * @note The set_time() method of boundary_function is called internally, so if the function is time-dependent use
     * get_time() in your implementation of the function to obtain t.
     */
    template<int dim,typename Number = double>
    void interpolate_boundary_values (
            dealii::IndexSet space_relevant_dofs ,
            idealii::slab::DoFHandler<dim> &dof_handler ,
            const dealii::types::boundary_id boundary_component ,
            dealii::Function<dim,Number> &boundary_function ,
            std::shared_ptr<dealii::AffineConstraints<Number>> spacetime_constraints ,
            const dealii::ComponentMask &component_mask =
                    dealii::ComponentMask () )
    {
        auto space_constraints = std::make_shared<
                dealii::AffineConstraints<Number>> ();
        space_constraints->reinit ( space_relevant_dofs );
        dealii::Quadrature < 1 > quad_time (
                dof_handler.temporal ()->get_fe ( 0 ).get_unit_support_points () );
        dealii::FEValues < 1 > fev (
                dof_handler.temporal ()->get_fe ( 0 ) , quad_time ,
                dealii::update_quadrature_points );

        unsigned int n_space_dofs = dof_handler.n_dofs_space ();
        unsigned int time_dof = 0;
        std::vector < dealii::types::global_dof_index > local_indices (
                dof_handler.dofs_per_cell_time () );
        //loop over time cells instead
        for ( const auto &cell_time : dof_handler.temporal ()->active_cell_iterators () )
        {
            fev.reinit ( cell_time );

            cell_time->get_dof_indices ( local_indices );
            for ( unsigned int q = 0 ; q < quad_time.size () ; q++ )
            {
                space_constraints->clear ();
                time_dof = local_indices[q];
                boundary_function.set_time (
                        fev.quadrature_point ( q )[0] );
                dealii::VectorTools::interpolate_boundary_values (
                        *dof_handler.spatial () ,
                        boundary_component , boundary_function ,
                        *space_constraints , component_mask );
                //				space_constraints->print(std::cout);
                for ( auto id = space_relevant_dofs.begin () ;
                        id != space_relevant_dofs.end () ; id++ )
                {
                    //check if this is a constrained dof
                    if ( space_constraints->is_constrained ( *id ) )
                    {
                        const std::vector<
                        std::pair<
                        dealii::types::global_dof_index,
                        double>> *entries =
                                space_constraints->get_constraint_entries (
                                        *id );
                        spacetime_constraints->add_line (
                                *id + time_dof * n_space_dofs );
                        //non Dirichlet constraint
                        if ( entries->size () > 0 )
                        {
                            for ( auto entry : *entries )
                            {
                                std::cout << entry.first << ","
                                        << entry.second << std::endl;
                                spacetime_constraints->add_entry (
                                        *id + time_dof * n_space_dofs ,
                                        entry.first + time_dof
                                        * n_space_dofs ,
                                        entry.second );
                            }
                        }
                        else
                        {
                            spacetime_constraints->set_inhomogeneity (
                                    *id + time_dof * n_space_dofs ,
                                    space_constraints->get_inhomogeneity (
                                            *id ) );
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Compute space-time constraints on the solution corresponding to Hcurl
     * Dirichlet conditions.
     * This function iterates over all temporal degrees of freedom and sets the
     * boundary values of the corresponding subvector to the value of the specified
     * at the time of the temporal dof.
     *
     * @param dof_handler The space-time slab dof handler the vector is indexed on
     * @param boundary_component The boundary id corresponding to the Dirichlet conditions
     * @param boundary_function The function that describes the Dirichlet data
     * @param spacetime_constraints The constraints to add the boundary values to
     * @param component_mask A component mask to only apply boundary values to certain FE components
     *
     * @note The set_time() method of boundary_function is called internally, so if the function is time-dependent use
     * get_time() in your implementation of the function to obtain t.
     */
    template<int dim,typename Number>
    void
    project_boundary_values_curl_conforming_l2 (
            idealii::slab::DoFHandler<dim> &dof_handler ,
            unsigned int first_vector_component ,
            dealii::Function<dim,Number> &boundary_function ,
            const dealii::types::boundary_id boundary_component ,
            std::shared_ptr<dealii::AffineConstraints<Number>> spacetime_constraints );

    /**
     * @brief Get the spatial subvector of a specific temporal dof of the corresponding slab.
     *
     * @param spacetime_vector The vector containing the slab space-time values
     * @param space_vector The resulting vector of spatial values at dof_index
     * @param dof_index The index of the DoF to be extracted.
     */
    void extract_subvector_at_time_dof (
            const dealii::Vector<double> &spacetime_vector ,
            dealii::Vector<double> &space_vector ,
            unsigned int dof_index )
    {
        unsigned int n_dofs_space = space_vector.size ();
        for ( unsigned int i = 0 ; i < n_dofs_space ; i++ )
        {
            space_vector[i] = spacetime_vector[i
                                               + dof_index * n_dofs_space];
        }
    }

    /**
     * @brief Get the spatial Trilinos subvector of a specific temporal dof of the corresponding slab.
     *
     * @param spacetime_vector The Trilinos vector containing the slab space-time values
     * @param space_vector The resulting Trilinos vector of spatial values at dof_index
     * @param dof_index The index of the DoF to be extracted.
     */
    void extract_subvector_at_time_dof (
            const dealii::TrilinosWrappers::MPI::Vector &spacetime_vector ,
            dealii::TrilinosWrappers::MPI::Vector &space_vector ,
            unsigned int dof_index )
    {
        dealii::IndexSet space_owned =
                space_vector.locally_owned_elements ();
        unsigned int n_dofs_space = space_vector.size ();
        dealii::TrilinosWrappers::MPI::Vector tmp;
        tmp.reinit ( space_owned ,
                     space_vector.get_mpi_communicator () );
        for ( auto id = space_owned.begin () ;
                id != space_owned.end () ; id++ )
        {
            tmp[*id] = spacetime_vector[*id + dof_index * n_dofs_space];
        }
        space_vector = tmp;
    }

    /**
     * @brief Get the spatial subvector at a specific time point of the corresponding slab.
     * The result is calculated by linear combination of each temporal dof vector according
     * to the underlying temporal finite element.
     * @note If t is not in the temporal interval of the slab, the resulting vector will be 0.
     * @param dof_handler The DoFHandler used to calculate the spacetime_vector.
     * @param spacetime_vector The vector containing the slab space-time values.
     * @param space_vector The resulting vector of spatial values at dof_index.
     * @param t The time point to extract from.
     */
    template<int dim>
    void extract_subvector_at_time_point (
            slab::DoFHandler<dim> &dof_handler ,
            const dealii::Vector<double> &spacetime_vector ,
            dealii::Vector<double> &space_vector , const double t )
    {
        space_vector = 0;
        unsigned int n_dofs_space = space_vector.size ();
        double left = 0;
        double right = 0;
        for ( auto cell : dof_handler.temporal ()->active_cell_iterators () )
        {
            left = cell->face ( 0 )->center () ( 0 );
            right = cell->face ( 1 )->center () ( 0 );
            if ( t < left || t > right )
            {
                continue;
            }
            double _t = ( t - left ) / ( right - left );
            dealii::Point < 1 > qpoint ( _t );
            const dealii::Quadrature<1> time_qf ( qpoint );
            dealii::FEValues < 1 > time_values (
                    dof_handler.temporal ()->get_fe () , time_qf ,
                    dealii::update_values );
            time_values.reinit ( cell );
            unsigned int offset = cell->index ()
                                                * dof_handler.dofs_per_cell_time ()
                                                * n_dofs_space;
            for ( unsigned int ii = 0 ;
                    ii < dof_handler.dofs_per_cell_time () ; ii++ )
            {
                double factor = time_values.shape_value ( ii , 0 );
                for ( unsigned int i = 0 ; i < n_dofs_space ; i++ )
                {
                    space_vector[i] += spacetime_vector[i
                                                        + ii * n_dofs_space + offset]
                                                        * factor;
                }
            }
        }
    }

    /**
     * @brief Get the spatial Trilinos subvector at a specific time point of the corresponding slab.
     * The result is calculated by linear combination of each temporal dof vector according
     * to the underlying temporal finite element.
     * @note If t is not in the temporal interval of the slab, the resulting vector will be 0.
     * @param dof_handler The DoFHandler used to calculate the spacetime_vector.
     * @param spacetime_vector The Trilinos vector containing the slab space-time values.
     * @param space_vector The resulting Trilinos vector of spatial values at dof_index.
     * @param t The time point to extract from.
     */
    template<int dim>
    void extract_subvector_at_time_point (
            slab::DoFHandler<dim> &dof_handler ,
            const dealii::TrilinosWrappers::MPI::Vector &spacetime_vector ,
            dealii::TrilinosWrappers::MPI::Vector &space_vector ,
            const double t )
    {
        space_vector = 0;
        unsigned int n_dofs_space = space_vector.size ();
        double left = 0;
        double right = 0;
        for ( auto cell : dof_handler.temporal ()->active_cell_iterators () )
        {
            left = cell->face ( 0 )->center () ( 0 );
            right = cell->face ( 1 )->center () ( 0 );
            if ( t < left || t > right )
            {
                continue;
            }
            double _t = ( t - left ) / ( right - left );
            dealii::Point < 1 > qpoint ( _t );
            const dealii::Quadrature<1> time_qf ( qpoint );
            dealii::FEValues < 1 > time_values (
                    dof_handler.temporal ()->get_fe () , time_qf ,
                    dealii::update_values );
            time_values.reinit ( cell );
            dealii::IndexSet space_owned =
                    space_vector.locally_owned_elements ();
            unsigned int n_dofs_space = space_vector.size ();
            dealii::TrilinosWrappers::MPI::Vector tmp;
            tmp.reinit ( space_owned ,
                         space_vector.get_mpi_communicator () );

            for ( unsigned int ii = 0 ;
                    ii < dof_handler.dofs_per_cell_time () ; ii++ )
            {
                double factor = time_values.shape_value ( ii , 0 );
                for ( auto id = space_owned.begin () ;
                        id != space_owned.end () ; id++ )
                {
                    tmp[*id] += spacetime_vector[*id
                                                 + ii * n_dofs_space]
                                                 * factor;
                }
                space_vector = tmp;
            }
        }
    }

    /**
     * @brief calculate the L2 inner product of (u-u_{kh}) with itself
     * @parameter dof_handler The slab::DoFHandler describing the dof distribution of the space-time vector
     * @parameter spacetime_vector The approximate solution u_{kh}
     * @parameter exact_solution The function describing the exact solution
     * @parameter quad The quadrature formula to use when calculating the integrals.
     */
    template<int dim>
    double
    calculate_L2L2_squared_error_on_slab (
            slab::DoFHandler<dim> &dof_handler ,
            dealii::Vector<double> &spacetime_vector ,
            dealii::Function<dim,double> &exact_solution ,
            spacetime::Quadrature<dim> &quad );
}

#endif /* INCLUDE_IDEAL_II_VECTOR_TOOLS_HH_ */
