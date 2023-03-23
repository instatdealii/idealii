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

////////////////////////////////////////////
// ideal.II includes
////////////////////////////////////////////

//same as in step-1
#include <ideal.II/base/time_iterator.hh>
#include <ideal.II/base/quadrature_lib.hh>
#include <ideal.II/dofs/slab_dof_tools.hh>
#include <ideal.II/dofs/spacetime_dof_handler.hh>
#include <ideal.II/fe/fe_dg.hh>
#include <ideal.II/fe/spacetime_fe_values.hh>
#include <ideal.II/grid/fixed_tria.hh>
#include <ideal.II/lac/spacetime_vector.hh>
#include <ideal.II/numerics/vector_tools.hh>

////////////////////////////////////////////
// deal.II includes
////////////////////////////////////////////

#include <deal.II/base/function.h>

// for extracting the number of velocity and pressure dofs
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
// For Stokes we have a system of finite elements
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
// for reading inp files and others
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

// needed to ensure the circle obstacle is refined correctly
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>

////////////////////////////////////////////
// C++ includes
////////////////////////////////////////////
#include <fstream>

/**
 * This function describes the Dirichlet data for the inflow boundary.
 * Note the use of get_time to obtain t.
 */
class PoisseuilleInflow : public dealii::Function<2>
{
public:
    PoisseuilleInflow ( double max_inflow_velocity = 1.5 ,
                        double channel_height = 0.41 )
    :
        Function<2> ( 3 ), max_inflow_velocity ( max_inflow_velocity ), H (
                channel_height )
                {
                }

    virtual double
    value ( const dealii::Point<2> &p ,
            const unsigned int component = 0 ) const;

    virtual void
    vector_value ( const dealii::Point<2> &p ,
                   dealii::Vector<double> &value ) const;
private:
    double max_inflow_velocity;
    double H;
};

double PoisseuilleInflow::value ( const dealii::Point<2> &p ,
                                  const unsigned int component ) const
{
    Assert ( component < this->n_components ,
             dealii::ExcIndexRange ( component , 0 , this->n_components ) )
            if ( component == 0 )
            {
                double y = p ( 1 );
                if ( p ( 0 ) == 0 && y <= 0.41 )
                {
                    double t = get_time ();
                    return 4 * max_inflow_velocity * y * ( H - y )
                            * std::sin ( M_PI * t * 0.125 )
                            / ( H * H );
                }
            }
    return 0;
}

void PoisseuilleInflow::vector_value ( const dealii::Point<2> &p ,
                                       dealii::Vector<double> &values ) const
{
    for ( unsigned int c = 0 ; c < this->n_components ; c++ )
    {
        values ( c ) = value ( p , c );
    }
}

//note that the exact solution is unknown and the force term (rhs) is zero.

// This class describes the solution of Stokes equations with
// space-time slab tensor-product elements
class Step2
{
public:
    Step2 ( unsigned int temporal_degree = 1 );
    void
    run ();

private:
    //nothing new here
    void
    make_grid ();
    void
    time_marching ();
    void
    setup_system_on_slab ();
    void
    assemble_system_on_slab ();
    void
    solve_system_on_slab ();
    void
    output_results_on_slab ();

    /////////////////////////////////////////////
    // space-time collections of slab objects
    /////////////////////////////////////////////
    idealii::spacetime::fixed::Triangulation<2> triangulation;
    idealii::spacetime::DoFHandler<2> dof_handler;
    idealii::spacetime::Vector<double> solution;

    // The space-time finite element description
    idealii::spacetime::DG_FiniteElement<2> fe;

    ////////////////////////////////////////////
    // objects needed on a single slab
    ////////////////////////////////////////////
    dealii::SparsityPattern sparsity_pattern;
    std::shared_ptr<dealii::AffineConstraints<double>> slab_constraints;
    dealii::SparseMatrix<double> slab_system_matrix;
    dealii::Vector<double> slab_system_rhs;
    dealii::Vector<double> slab_initial_value;
    unsigned int slab;

    ///////////////////////////////////////////////////////////////
    // Struct holding all iterators over the space-time objects.
    ///////////////////////////////////////////////////////////////
    struct
    {
        idealii::slab::TriaIterator<2> tria;
        idealii::slab::DoFHandlerIterator<2> dof;
        idealii::slab::VectorIterator<double> solution;
    } slab_its;
};

Step2::Step2 ( unsigned int temporal_degree )
:
                triangulation (), dof_handler ( &triangulation ),
                // space-time finite element with
                // * continuous Taylor-Hood i.e. Q2/Q1 element in space
                //     vector-valued biquadratic velocity
                //     scalar-valued bilinear pressure
                // * discontinuous Lagrangian FE_DGQ finite element in time
                fe ( std::make_shared < dealii::FESystem
                     < 2 >> ( dealii::FE_Q < 2 > ( 2 ) , 2 , dealii::FE_Q < 2 > ( 1 ) , 1 ) ,
                     temporal_degree ), slab ( 0 )
{
}

void Step2::run ()
{
    make_grid ();
    time_marching ();
}

void Step2::make_grid ()
{
    auto space_tria = std::make_shared<dealii::Triangulation<2>> ();
    // instead of a grid generator we read the mesh from the provided input file
    dealii::GridIn < 2 > grid_in;
    grid_in.attach_triangulation ( *space_tria );
    std::ifstream input_file ( "nsbench4.inp" );
    grid_in.read_ucd ( input_file );
    // The interior obstacle is a circle in theory
    // In the input mesh it is approximated by a polyhedron and so
    // we need to tell the triangulation to properly refine it.
    dealii::Point < 2 > p ( 0.2 , 0.2 );
    static const dealii::SphericalManifold<2> boundary ( p );
    dealii::GridTools::copy_boundary_to_manifold_id ( *space_tria );
    space_tria->set_manifold ( 80 , boundary );

    const unsigned int M = 32;
    triangulation.generate ( space_tria , M );
    triangulation.refine_global ( 2 , 0 );
    dof_handler.generate ();
}

void Step2::time_marching ()
{
    //This function is the same as before
    idealii::TimeIteratorCollection < 2 > tic = idealii::TimeIteratorCollection<
            2> ();

    solution.reinit ( triangulation.M () );

    slab_its.tria = triangulation.begin ();
    slab_its.dof = dof_handler.begin ();
    slab_its.solution = solution.begin ();
    slab = 0;
    slab_initial_value = 0;	//initial value is 0 for now
    tic.add_iterator ( &slab_its.tria , &triangulation );
    tic.add_iterator ( &slab_its.dof , &dof_handler );
    tic.add_iterator ( &slab_its.solution , &solution );
    std::cout << "*******Starting time-stepping*********" << std::endl;
    for ( ; !tic.at_end () ; tic.increment () )
    {
        std::cout << "Starting time-step (" << slab_its.tria->startpoint ()
                << "," << slab_its.tria->endpoint () << "]" << std::endl;

        setup_system_on_slab ();
        std::cout << "setup done" << std::endl;
        assemble_system_on_slab ();
        std::cout << "assembly done" << std::endl;
        solve_system_on_slab ();
        output_results_on_slab ();
        idealii::slab::VectorTools::extract_subvector_at_time_dof (
                *slab_its.solution , slab_initial_value ,
                slab_its.tria->temporal ()->n_global_active_cells () - 1 );
        slab++;
    }
}

void Step2::setup_system_on_slab ()
{
    slab_its.dof->distribute_dofs ( fe );

    dealii::DoFRenumbering::component_wise ( *slab_its.dof->spatial () );

    std::cout << "Number of degrees of freedom: \n\t"
            << slab_its.dof->n_dofs_space () << " (space) * "
            << slab_its.dof->n_dofs_time () << " (time) = "
            << slab_its.dof->n_dofs_spacetime () << " (spacetime)" << std::endl;
    if ( slab == 0 )
    {
        slab_initial_value.reinit ( slab_its.dof->n_dofs_space () );
        slab_initial_value = 0;
    }

    slab_constraints = std::make_shared<dealii::AffineConstraints<double>> ();

    // Now we have multiple Dirichlet boundaries.
    // 0 is assigned to the left wall as an inhom. inflow
    // 2 is assigned to the upper and lower wall as no-slip (hom.)
    // 80 is assigned to the obstacle as no-slip as well
    auto zero = dealii::Functions::ZeroFunction < 2 > ( 3 );
    auto inflow = PoisseuilleInflow ();
    std::vector<bool> component_mask ( 3 , true );
    component_mask[2] = false;

    idealii::slab::VectorTools::interpolate_boundary_values ( *slab_its.dof ,
                                                              0 , inflow ,
                                                              slab_constraints ,
                                                              component_mask );

    idealii::slab::VectorTools::interpolate_boundary_values ( *slab_its.dof ,
                                                              2 , zero ,
                                                              slab_constraints ,
                                                              component_mask );

    idealii::slab::VectorTools::interpolate_boundary_values ( *slab_its.dof ,
                                                              80 , zero ,
                                                              slab_constraints ,
                                                              component_mask );

    slab_constraints->close ();

    // The spatial pressure-pressure coupling block is empty
    // and the sparsity pattern can be empty in these blocks
    dealii::Table < 2 , dealii::DoFTools::Coupling
    > coupling_space ( 2 + 1 , 2 + 1 );
    // init with no coupling
    coupling_space.fill ( dealii::DoFTools::none );
    // coupling of velocities
    for ( unsigned int i = 0 ; i < 2 ; i++ )
    {
        for ( unsigned int j = 0 ; j < 2 ; j++ )
        {
            coupling_space[i][j] = dealii::DoFTools::always;
        }
    }
    // coupling of velocity and pressure
    for ( unsigned int i = 0 ; i < 2 ; i++ )
    {
        coupling_space[i][2] = dealii::DoFTools::always;
        coupling_space[2][i] = dealii::DoFTools::always;
    }

    dealii::DynamicSparsityPattern dsp ( slab_its.dof->n_dofs_spacetime () );
    idealii::slab::DoFTools::make_upwind_sparsity_pattern ( *slab_its.dof ,
                                                            coupling_space ,
                                                            dsp ,
                                                            slab_constraints );
    sparsity_pattern.copy_from ( dsp );
    slab_system_matrix.reinit ( sparsity_pattern );

    slab_its.solution->reinit ( slab_its.dof->n_dofs_spacetime () );
    slab_system_rhs.reinit ( slab_its.dof->n_dofs_spacetime () );
}

void Step2::assemble_system_on_slab ()
{
    idealii::spacetime::QGauss < 2 > quad ( fe.spatial ()->degree + 3 ,
                                            fe.temporal ()->degree + 1 );

    idealii::spacetime::FEValues < 2 > fe_values_spacetime (
            fe ,
            quad ,
            dealii::update_values | dealii::update_gradients
            | dealii::update_quadrature_points | dealii::update_JxW_values );

    idealii::spacetime::FEJumpValues < 2 > fe_jump_values_spacetime (
            fe ,
            quad ,
            dealii::update_values | dealii::update_gradients
            | dealii::update_quadrature_points | dealii::update_JxW_values );

    const unsigned int dofs_per_spacetime_cell = fe.dofs_per_cell;

    auto N = slab_its.tria->temporal ()->n_global_active_cells ();
    dealii::FullMatrix<double> cell_matrix ( N * dofs_per_spacetime_cell ,
                                             N * dofs_per_spacetime_cell );
    dealii::Vector<double> cell_rhs ( N * dofs_per_spacetime_cell );

    std::vector < dealii::types::global_dof_index > local_spacetime_dof_index (
            N * dofs_per_spacetime_cell );

    unsigned int n;
    unsigned int n_quad_spacetime = fe_values_spacetime.n_quadrature_points;
    unsigned int n_quad_space = quad.spatial ()->size ();

    // To get the velocity and pressure subvalues of the fe system shape functions
    // we use the following extractors
    dealii::FEValuesExtractors::Vector velocity ( 0 );
    dealii::FEValuesExtractors::Scalar pressure ( 2 );
    // set the kinematic viscosity
    double nu_f = 1.0e-3;

    for ( const auto &cell_space : slab_its.dof->spatial ()->active_cell_iterators () )
    {
        fe_values_spacetime.reinit_space ( cell_space );
        fe_jump_values_spacetime.reinit_space ( cell_space );
        std::vector<dealii::Tensor<1,2>> initial_values (
                fe_values_spacetime.spatial ()->n_quadrature_points );
        fe_values_spacetime.spatial ()->operator[] ( velocity ).get_function_values (
                slab_initial_value , initial_values );

        cell_matrix = 0;
        cell_rhs = 0;
        for ( const auto &cell_time : slab_its.dof->temporal ()->active_cell_iterators () )
        {
            n = cell_time->index ();
            fe_values_spacetime.reinit_time ( cell_time );
            fe_jump_values_spacetime.reinit_time ( cell_time );
            fe_values_spacetime.get_local_dof_indices (
                    local_spacetime_dof_index );

            for ( unsigned int q = 0 ; q < n_quad_spacetime ; ++q )
            {

                for ( unsigned int i = 0 ; i < dofs_per_spacetime_cell ; ++i )
                {
                    for ( unsigned int j = 0 ; j < dofs_per_spacetime_cell ;
                            ++j )
                    {

                        // The FEValues calls with extractors are a bit different to the ones in deal.II.
                        // There, all possibilities are saved to a cache, so duplicating this cache
                        // is unnecessarily ineffective.
                        // Instead of the operator[]() returning a Scalar or Vector FEValues object
                        // depending on the extractor we provide different functions for
                        // vector and scalar valued shape functions.
                        // For example
                        // vector_value(extractor,i,q) instead of [extractor].value(i,q);
                        // See Readme.md for the list of corresponding shape functions
                        // used here.

                        // (dt u, v)
                        cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                      j + n * dofs_per_spacetime_cell ) +=
                                              fe_values_spacetime.vector_value ( velocity ,
                                                                                 i , q )
                                                                                 * fe_values_spacetime.vector_dt ( velocity , j ,
                                                                                                                   q )
                                                                                                                   * fe_values_spacetime.JxW ( q );

                        // (grad u, grad v)
                        cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                      j + n * dofs_per_spacetime_cell )
                        // as the shape function is vector valued we need to take the
                        // scalar product.
                        += dealii::scalar_product (
                                fe_values_spacetime.vector_space_grad (
                                        velocity , i , q ) ,
                                        fe_values_spacetime.vector_space_grad (
                                                velocity , j , q ) )
                                                * fe_values_spacetime.JxW ( q ) * nu_f;

                        // (pressure gradient)
                        cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                      j + n * dofs_per_spacetime_cell ) -=
                                              fe_values_spacetime.vector_divergence (
                                                      velocity , i , q )
                                                      * fe_values_spacetime.scalar_value ( pressure ,
                                                                                           j , q )
                                                                                           * fe_values_spacetime.JxW ( q );
                        // (div free constraint)
                        cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                      j + n * dofs_per_spacetime_cell ) +=
                                              fe_values_spacetime.scalar_value ( pressure ,
                                                                                 i , q )
                                                                                 * fe_values_spacetime.vector_divergence (
                                                                                         velocity , j , q )
                                                                                         * fe_values_spacetime.JxW ( q );
                    } //dofs j
                } //dofs i
            } //quad

            // Jump terms and initial values just have a spatial loop
            // Only the velocity has a temporal derivative, so we don't need
            // jump values for the pressure.
            for ( unsigned int q = 0 ; q < n_quad_space ; ++q )
            {
                for ( unsigned int i = 0 ; i < dofs_per_spacetime_cell ; ++i )
                {
                    for ( unsigned int j = 0 ; j < dofs_per_spacetime_cell ;
                            ++j )
                    {
                        // (phi^+, v^+)
                        cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                      j + n * dofs_per_spacetime_cell ) +=
                                              fe_jump_values_spacetime.vector_value_plus (
                                                      velocity , i , q )
                                                      * fe_jump_values_spacetime.vector_value_plus (
                                                              velocity , j , q )
                                                              * fe_jump_values_spacetime.JxW ( q );

                        // -(phi^+, v^-)
                        if ( n > 0 )
                        {
                            cell_matrix (
                                    i + n * dofs_per_spacetime_cell ,
                                    j + ( n - 1 ) * dofs_per_spacetime_cell ) -=
                                            fe_jump_values_spacetime.vector_value_plus (
                                                    velocity , i , q )
                                                    * fe_jump_values_spacetime.vector_value_minus (
                                                            velocity , j , q )
                                                            * fe_jump_values_spacetime.JxW ( q );
                        }
                    } //dofs j
                    if ( n == 0 )
                    {
                        cell_rhs ( i ) +=
                                fe_jump_values_spacetime.vector_value_plus (
                                        velocity , i , q )
                                        * initial_values[q] *
                                        //value of previous solution at t0
                                        fe_jump_values_spacetime.JxW ( q );
                    }
                } //dofs i
            }

        } //cell time
        slab_constraints->distribute_local_to_global (
                cell_matrix , cell_rhs , local_spacetime_dof_index ,
                slab_system_matrix , slab_system_rhs );
    } //cell space
}

void Step2::solve_system_on_slab ()
{
    dealii::SparseDirectUMFPACK solver;
    solver.factorize ( slab_system_matrix );
    solver.vmult ( *slab_its.solution , slab_system_rhs );
    slab_constraints->distribute ( *slab_its.solution );
}

void Step2::output_results_on_slab ()
{
    auto n_dofs = slab_its.dof->n_dofs_time ();
    // As in deal.II we need to tell the DataOut what to do with the vector and scalar valued
    // entries in the solution vector.
    std::vector < std::string > field_names;
    std::vector < dealii::DataComponentInterpretation::DataComponentInterpretation > dci;
    for ( unsigned int i = 0 ; i < 2 ; i++ )
    {
        field_names.push_back ( "velocity" );
        dci.push_back (
                dealii::DataComponentInterpretation::component_is_part_of_vector );
    }
    field_names.push_back ( "pressure" );
    dci.push_back ( dealii::DataComponentInterpretation::component_is_scalar );

    for ( unsigned i = 0 ; i < n_dofs ; i++ )
    {
        dealii::DataOut < 2 > data_out;
        data_out.attach_dof_handler ( *slab_its.dof->spatial () );
        dealii::Vector<double> local_solution;
        local_solution.reinit ( slab_its.dof->n_dofs_space () );
        idealii::slab::VectorTools::extract_subvector_at_time_dof (
                *slab_its.solution , local_solution , i );
        data_out.add_data_vector ( local_solution , field_names ,
                                   dealii::DataOut < 2 > ::type_dof_data ,
                                   dci );
        data_out.build_patches ( 1 );
        std::ostringstream filename;
        filename << "solution_dG(" << fe.temporal ()->degree << ")_t_"
                << slab * n_dofs + i << ".vtk";

        std::ofstream output ( filename.str () );
        data_out.write_vtk ( output );
        output.close ();
    }
}

int main ()
{
    Step2 problem ( 1 );
    problem.run ();
}
