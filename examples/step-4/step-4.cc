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

////////////////////////////////////////////////////////////////////////
// WORK IN PROGRESS!
// There is some problem with building the matrix or rhs for multi-element
// slabs. Until this works, the example is limited to single element slabs
// i.e. time-stepping style
////////////////////////////////////////////////////////////////////////

// number of thread parallel threads, not used so far
// but extension to dealii::WorkStream would be possible
#define MPIX_THREADS 1

////////////////////////////////////////////
// ideal.II includes
////////////////////////////////////////////

#include <ideal.II/distributed/fixed_tria.hh>
#include <ideal.II/lac/spacetime_trilinos_vector.hh>

//all other ideal.II includes are known
#include <ideal.II/base/time_iterator.hh>
#include <ideal.II/base/quadrature_lib.hh>
#include <ideal.II/dofs/spacetime_dof_handler.hh>
#include <ideal.II/dofs/slab_dof_tools.hh>
#include <deal.II/dofs/dof_renumbering.h>
#include <ideal.II/fe/fe_dg.hh>
#include <ideal.II/fe/spacetime_fe_values.hh>
#include <ideal.II/numerics/vector_tools.hh>

////////////////////////////////////////////
// deal.II includes
////////////////////////////////////////////

#include <deal.II/distributed/tria.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>

////////////////////////////////////////////
// C++ includes
////////////////////////////////////////////
#include <fstream>

/**
 * This function describes the Dirichlet data for the inflow boundary.
 */
class PoisseuilleInflow : public dealii::Function<2>
{
public:
    PoisseuilleInflow ( double max_inflow_velocity = 1.5 , double channel_height = 0.41 )
        :
        Function<2> ( 3 ),
        max_inflow_velocity ( max_inflow_velocity ),
        H ( channel_height )
    {
    }

    virtual double
    value ( const dealii::Point<2> &p , const unsigned int component = 0 ) const;

    virtual void
    vector_value ( const dealii::Point<2> &p , dealii::Vector<double> &value ) const;
    private:
    double max_inflow_velocity;
    double H;
};

double PoisseuilleInflow::value ( const dealii::Point<2> &p , const unsigned int component ) const
{
    Assert ( component < this->n_components , dealii::ExcIndexRange ( component , 0 , this->n_components ) )
    if ( component == 0 )
    {
        double y = p ( 1 );
        if ( p ( 0 ) == 0 && y <= 0.41 )
        {
            double t = get_time ();
            return 4 * max_inflow_velocity * y * ( H - y ) * std::sin ( M_PI * t * 0.125 ) / ( H * H );
        }
    }
    return 0;
}

void PoisseuilleInflow::vector_value ( const dealii::Point<2> &p , dealii::Vector<double> &values ) const
{
    for ( unsigned int c = 0 ; c < this->n_components ; c++ )
    {
        values ( c ) = value ( p , c );
    }
}

// note that the exact solution is unknown and the force term (rhs) is zero.

// This class describes the solution of the nonlinear Navier-Stokes equation with
// space-time slab tensor-product elements and Newton linearization
class Step4
{
public:
    Step4 ( unsigned int temporal_degree );
    void
    run ();

private:
    void
    make_grid ();
    void
    time_marching ();
    void
    setup_system_on_slab ();
    void
    assemble_residual_on_slab ();
    void
    assemble_system_on_slab ();
    void
    solve_Newton_problem_on_slab ();
    void
    output_results_on_slab ();

    // we need to know the MPI communicator
    MPI_Comm mpi_comm;

    dealii::ConditionalOStream pout;
    /////////////////////////////////////////////
    // space-time collections of slab objects
    /////////////////////////////////////////////
    idealii::spacetime::parallel::distributed::fixed::Triangulation<2> triangulation;
    idealii::spacetime::DoFHandler<2> dof_handler;
    idealii::spacetime::TrilinosVector solution;

    // The space-time finite element description
    idealii::spacetime::DG_FiniteElement<2> fe;

    ////////////////////////////////////////////
    // objects needed on a single slab
    ////////////////////////////////////////////
    dealii::SparsityPattern slab_sparsity_pattern;
    std::shared_ptr<dealii::AffineConstraints<double>> slab_zero_constraints;
    dealii::TrilinosWrappers::SparseMatrix slab_system_matrix;
    dealii::TrilinosWrappers::MPI::Vector slab_system_rhs;
    dealii::TrilinosWrappers::MPI::Vector slab_initial_value;
    unsigned int slab;

    dealii::TrilinosWrappers::MPI::Vector slab_newton_update;
    dealii::TrilinosWrappers::MPI::Vector slab_owned_tmp;
    dealii::TrilinosWrappers::MPI::Vector slab_relevant_tmp;

    dealii::IndexSet slab_locally_owned_dofs;
    dealii::IndexSet slab_locally_relevant_dofs;

    dealii::IndexSet space_locally_owned_dofs;
    dealii::IndexSet space_locally_relevant_dofs;

    ////////////////////////////////////////////////////////////
    // Struct holding all iterators over the space-time objects.
    ////////////////////////////////////////////////////////////
    struct
    {
        idealii::slab::parallel::distributed::TriaIterator<2> tria;
        idealii::slab::DoFHandlerIterator<2> dof;
        idealii::slab::TrilinosVectorIterator solution;
    } slab_its;
};

Step4::Step4 ( unsigned int temporal_degree )
    :
    mpi_comm ( MPI_COMM_WORLD ),
    pout ( std::cout , dealii::Utilities::MPI::this_mpi_process ( mpi_comm ) == 0 ),
    triangulation (),
    dof_handler ( &triangulation ),
    fe ( std::make_shared < dealii::FESystem< 2 >>
             ( dealii::FE_Q < 2 > ( 2 ) , 2 , dealii::FE_Q < 2 > ( 1 ) , 1 ) ,
         temporal_degree,
         idealii::spacetime::DG_FiniteElement<2>::support_type::Legendre
    ),
    slab ( 0 )
{
}

void Step4::run ()
{
    make_grid ();
    time_marching ();
}

void Step4::make_grid ()
{
    //construct an MPI parallel triangulation with the provided MPI communicator
    auto space_tria = std::make_shared < dealii::parallel::distributed::Triangulation < 2 >> ( mpi_comm );
    dealii::GridIn < 2 > grid_in;
    grid_in.attach_triangulation ( *space_tria );
    std::ifstream input_file ( "nsbench4.inp" );
    grid_in.read_ucd ( input_file );
    dealii::Point < 2 > p ( 0.2 , 0.2 );
    static const dealii::SphericalManifold<2> boundary ( p );
    dealii::GridTools::copy_boundary_to_manifold_id ( *space_tria );
    space_tria->set_manifold ( 80 , boundary );
    const unsigned int M = 256;
    triangulation.generate ( space_tria , M , 0 , 8. );
    triangulation.refine_global ( 2 , 0 );
    dof_handler.generate ();
}

void Step4::time_marching ()
{
    idealii::TimeIteratorCollection < 2 > tic = idealii::TimeIteratorCollection<2> ();

    solution.reinit ( triangulation.M () );

    slab_its.tria = triangulation.begin ();
    slab_its.dof = dof_handler.begin ();
    slab_its.solution = solution.begin ();
    slab = 0;
    tic.add_iterator ( &slab_its.tria , &triangulation );
    tic.add_iterator ( &slab_its.dof , &dof_handler );
    tic.add_iterator ( &slab_its.solution , &solution );
    pout << "*******Starting time-stepping*********" << std::endl;
    for ( ; !tic.at_end () ; tic.increment () )
    {
        pout << "Starting time-step ("
             << slab_its.tria->startpoint () << ","
             << slab_its.tria->endpoint () << ")"
             << std::endl;

        pout << "setup system" << std::endl;
        setup_system_on_slab ();
        solve_Newton_problem_on_slab ();
        output_results_on_slab ();
        // As in step-3 we need to extract the solution at the final time point as the
        // final DoF is not in the same location for Legendre support points
        idealii::slab::VectorTools::extract_subvector_at_time_point ( *slab_its.dof ,
                                                                      *slab_its.solution ,
                                                                      slab_initial_value ,
                                                                      slab_its.tria->endpoint () );
        slab++;
    }
}

void Step4::setup_system_on_slab ()
{
    slab_its.dof->distribute_dofs ( fe );

    dealii::DoFRenumbering::component_wise ( *slab_its.dof->spatial () );

    pout << "Number of degrees of freedom: \n\t"
         << slab_its.dof->n_dofs_space () << " (space) * "
         << slab_its.dof->n_dofs_time ()  << " (time) = "
         << slab_its.dof->n_dofs_spacetime () << std::endl;

    // we need to know the spatial set of degrees of freedom owned by the current MPI processor
    space_locally_owned_dofs = slab_its.dof->spatial ()->locally_owned_dofs ();
    // and beloging to elements owned by the processor
    dealii::DoFTools::extract_locally_relevant_dofs ( *slab_its.dof->spatial () , space_locally_relevant_dofs );

    // The same holds for the set of space-time degrees of freedom
    slab_locally_owned_dofs = slab_its.dof->locally_owned_dofs ();
    slab_locally_relevant_dofs = idealii::slab::DoFTools::extract_locally_relevant_dofs ( *slab_its.dof );

    slab_owned_tmp.reinit ( slab_locally_owned_dofs , mpi_comm );
    slab_relevant_tmp.reinit ( slab_locally_owned_dofs , slab_locally_relevant_dofs , mpi_comm );
    slab_newton_update.reinit ( slab_locally_owned_dofs , mpi_comm );
    // On the first slab the initial value has to be set to the correct set of spatially relevant
    // dofs
    if ( slab == 0 )
    {
        slab_initial_value.reinit ( space_locally_owned_dofs , space_locally_relevant_dofs , mpi_comm );
        slab_initial_value = 0;
    }

    // We need two sets of constraints
    // The correct boundary conditions are prescribed on the initial Newton guess
    // and so all Dirichlet boundary conditions on the Newton update need to be
    // zero to keep the solution unchanged at Gamma_D
    slab_zero_constraints = std::make_shared<dealii::AffineConstraints<double>> ();
    auto slab_initial_constraints = std::make_shared<dealii::AffineConstraints<double>> ();
    auto zero = dealii::Functions::ZeroFunction < 2 > ( 3 );
    auto inflow = PoisseuilleInflow ();
    std::vector<bool> component_mask ( 3 , true );
    component_mask[2] = false;

    idealii::slab::VectorTools::interpolate_boundary_values ( *slab_its.dof ,
                                                              0 ,
                                                              inflow ,
                                                              slab_initial_constraints ,
                                                              component_mask );

    idealii::slab::VectorTools::interpolate_boundary_values ( *slab_its.dof ,
                                                              2 ,
                                                              zero ,
                                                              slab_initial_constraints ,
                                                              component_mask );

    idealii::slab::VectorTools::interpolate_boundary_values ( *slab_its.dof ,
                                                              80 ,
                                                              zero ,
                                                              slab_initial_constraints ,
                                                              component_mask );

    slab_initial_constraints->close ();

    idealii::slab::VectorTools::interpolate_boundary_values ( *slab_its.dof ,
                                                              0 ,
                                                              zero ,
                                                              slab_zero_constraints ,
                                                              component_mask );

    idealii::slab::VectorTools::interpolate_boundary_values ( *slab_its.dof ,
                                                              2 ,
                                                              zero ,
                                                              slab_zero_constraints ,
                                                              component_mask );

    idealii::slab::VectorTools::interpolate_boundary_values ( *slab_its.dof ,
                                                              80 ,
                                                              zero ,
                                                              slab_zero_constraints ,
                                                              component_mask );

    slab_zero_constraints->close ();
    std::map<dealii::types::global_dof_index,double> initial_bc;

    dealii::IndexSet::ElementIterator lri = space_locally_owned_dofs.begin ();
    dealii::IndexSet::ElementIterator lre = space_locally_owned_dofs.end ();
    for ( ; lri != lre ; lri++ )
    {
        for ( unsigned int ii = 0 ; ii < slab_its.dof->n_dofs_time () ; ii++ )
        {
            slab_owned_tmp[*lri + slab_its.dof->n_dofs_space () * ii] = slab_initial_value[*lri];
        }
    }
    slab_initial_constraints->distribute ( slab_owned_tmp );

    // The spatial pressure-pressure coupling block is empty
    // and the sparsity pattern can be empty in these blocks
    dealii::Table < 2 , dealii::DoFTools::Coupling > coupling_space ( 2 + 1 , 2 + 1 );
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
    idealii::slab::DoFTools::make_upwind_sparsity_pattern ( *slab_its.dof , coupling_space , dsp , slab_zero_constraints );

    // To save memory we distribute the sparsity pattern to only hold the locally
    // needed set of space-time degrees of freedom
    dealii::SparsityTools::distribute_sparsity_pattern (
            dsp ,
            slab_locally_owned_dofs ,
            mpi_comm ,
            slab_locally_relevant_dofs );

    // reinit the system matrix and vectors to hold only local information
    slab_system_matrix.reinit ( slab_locally_owned_dofs , slab_locally_owned_dofs , dsp );

    slab_its.solution->reinit ( slab_locally_owned_dofs , slab_locally_relevant_dofs , mpi_comm );
    slab_system_rhs.reinit ( slab_locally_owned_dofs , mpi_comm );
    *slab_its.solution = slab_owned_tmp;
}

void Step4::assemble_system_on_slab ()
{

    slab_system_matrix = 0;
    idealii::spacetime::QGauss < 2 > quad ( fe.spatial ()->degree + 3 , fe.temporal ()->degree + 2 );

    idealii::spacetime::FEValues < 2 > fe_values_spacetime (fe , quad ,
            dealii::update_values |
            dealii::update_gradients |
            dealii::update_quadrature_points |
            dealii::update_JxW_values );

    idealii::spacetime::FEJumpValues < 2 > fe_jump_values_spacetime (fe , quad ,
            dealii::update_values |
            dealii::update_gradients |
            dealii::update_quadrature_points |
            dealii::update_JxW_values );

    const unsigned int dofs_per_spacetime_cell = fe.dofs_per_cell;

    auto N = slab_its.tria->temporal ()->n_global_active_cells ();
    dealii::FullMatrix<double> cell_matrix ( N * dofs_per_spacetime_cell , N * dofs_per_spacetime_cell );

    std::vector < dealii::types::global_dof_index > local_spacetime_dof_index ( N * dofs_per_spacetime_cell );

    unsigned int n;
    unsigned int n_quad_spacetime = fe_values_spacetime.n_quadrature_points;
    unsigned int n_quad_space = quad.spatial ()->size ();

    dealii::FEValuesExtractors::Vector velocity ( 0 );
    dealii::FEValuesExtractors::Scalar pressure ( 2 );
    // set the kinematic viscosity
    double nu_f = 1.0e-3;

    std::vector<dealii::Vector<double>> old_solution_values ( n_quad_spacetime , dealii::Vector<double> ( 3 ) );

    std::vector < std::vector<dealii::Tensor<1,2> > > old_solution_grads ( n_quad_spacetime ,
                                                                           std::vector<dealii::Tensor<1,2>> ( 3 ) );
    for ( const auto &cell_space : slab_its.dof->spatial ()->active_cell_iterators () )
    {
        // We only can calculate the contributions of processor local cells
        // Apart from that, only the additional convection term is new
        if ( cell_space->is_locally_owned () )
        {
            fe_values_spacetime.reinit_space ( cell_space );
            fe_jump_values_spacetime.reinit_space ( cell_space );

            cell_matrix = 0;
            for ( const auto &cell_time : slab_its.dof->temporal ()->active_cell_iterators () )
            {
                n = cell_time->index ();
                fe_values_spacetime.reinit_time ( cell_time );
                fe_jump_values_spacetime.reinit_time ( cell_time );
                fe_values_spacetime.get_local_dof_indices ( local_spacetime_dof_index );

                fe_values_spacetime.get_function_values( *slab_its.solution, old_solution_values );

                fe_values_spacetime.get_function_space_gradients( *slab_its.solution , old_solution_grads );
                for ( unsigned int q = 0 ; q < n_quad_spacetime ; ++q )
                {

                    dealii::Tensor < 1 , 2 > v;
                    dealii::Tensor < 2 , 2 > grad_v;
                    for ( int c = 0 ; c < 2 ; c++ )
                    {
                        v[c] = old_solution_values[q] ( c );
                        for ( int d = 0 ; d < 2 ; d++ )
                        {
                            grad_v[c][d] = old_solution_grads[q][c][d];
                        }
                    }
                    for ( unsigned int i = 0 ; i < dofs_per_spacetime_cell ; ++i )
                    {

                        for ( unsigned int j = 0 ; j < dofs_per_spacetime_cell ; ++j )
                        {

                            // (convection term)
                            cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                          j + n * dofs_per_spacetime_cell )
                            += fe_values_spacetime.vector_value ( velocity , i , q ) *
                               (
                                   fe_values_spacetime.vector_space_grad ( velocity , j , q ) * v
                                   + grad_v * fe_values_spacetime.vector_value ( velocity , j , q )
                               ) * fe_values_spacetime.JxW ( q );

                            // (phi, dt v)
                            cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                          j + n * dofs_per_spacetime_cell )
                            += fe_values_spacetime.vector_value ( velocity , i , q )
                             * fe_values_spacetime.vector_dt ( velocity , j , q )
                             * fe_values_spacetime.JxW ( q );

                            // (grad phi, grad v)
                            cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                          j + n * dofs_per_spacetime_cell )
                            += dealii::scalar_product (
                                    fe_values_spacetime.vector_space_grad ( velocity , i , q ) ,
                                    fe_values_spacetime.vector_space_grad ( velocity , j , q )
                                )
                                * fe_values_spacetime.JxW ( q ) * nu_f;

                            // (pressure gradient)
                            cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                          j + n * dofs_per_spacetime_cell )
                            -= fe_values_spacetime.vector_divergence ( velocity , i , q )
                             * fe_values_spacetime.scalar_value ( pressure , j , q )
                             * fe_values_spacetime.JxW ( q );

                            // (div free constraint)
                            cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                          j + n * dofs_per_spacetime_cell )
                            += fe_values_spacetime.scalar_value ( pressure , i , q )
                             * fe_values_spacetime.vector_divergence ( velocity , j , q )
                             * fe_values_spacetime.JxW ( q );

                        } //dofs j
                    } //dofs i
                } //quad

                for ( unsigned int q = 0 ; q < n_quad_space ; ++q )
                {
                    for ( unsigned int i = 0 ; i < dofs_per_spacetime_cell ; ++i )
                    {
                        for ( unsigned int j = 0 ; j < dofs_per_spacetime_cell ; ++j )
                        {
                            // (v^+, u^+)
                            cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                          j + n * dofs_per_spacetime_cell )
                              += fe_jump_values_spacetime.vector_value_plus ( velocity , i , q )
                               * fe_jump_values_spacetime.vector_value_plus ( velocity , j , q )
                               * fe_jump_values_spacetime.JxW ( q );

                            // -(v^-, u^+)
                            if ( n > 0 )
                            {
                                cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                              j + ( n - 1 ) * dofs_per_spacetime_cell )
                                -= fe_jump_values_spacetime.vector_value_plus ( velocity , i , q )
                                 * fe_jump_values_spacetime.vector_value_minus ( velocity , j , q )
                                 * fe_jump_values_spacetime.JxW ( q );
                            }
                        } //dofs j
                    } //dofs i
                }

            } //cell time
            slab_zero_constraints->distribute_local_to_global (
                    cell_matrix , local_spacetime_dof_index , slab_system_matrix );
        }
    } //cell space

    // We need to communicate local contributions to other processors after assembly
    slab_system_matrix.compress ( dealii::VectorOperation::add );
}

// For the Newton iteration we also need to assemble the residual vector
// this depends on the current slab solution/iterate.
// Currently evaluation of the solution is done by hand and functions to
// simplify this in ideal.II are work in progress
void Step4::assemble_residual_on_slab ()
{
    slab_system_rhs = 0;
    idealii::spacetime::QGauss < 2 > quad ( fe.spatial ()->degree + 3 , fe.temporal ()->degree + 2 );

    idealii::spacetime::FEValues < 2 > fe_values_spacetime (fe , quad ,
                                                            dealii::update_values |
                                                            dealii::update_gradients |
                                                            dealii::update_quadrature_points |
                                                            dealii::update_JxW_values );

    idealii::spacetime::FEJumpValues < 2 > fe_jump_values_spacetime (fe , quad ,
                                                                     dealii::update_values |
                                                                     dealii::update_gradients |
                                                                     dealii::update_quadrature_points |
                                                                     dealii::update_JxW_values );

    const unsigned int dofs_per_spacetime_cell = fe.dofs_per_cell;

    auto N = slab_its.tria->temporal ()->n_global_active_cells ();
    dealii::Vector<double> cell_rhs ( N * dofs_per_spacetime_cell );

    std::vector < dealii::types::global_dof_index > local_spacetime_dof_index ( N * dofs_per_spacetime_cell );

    unsigned int n;
    unsigned int n_quad_spacetime = fe_values_spacetime.n_quadrature_points;
    unsigned int n_quad_space = quad.spatial ()->size ();

    dealii::FEValuesExtractors::Vector velocity ( 0 );
    dealii::FEValuesExtractors::Scalar pressure ( 2 );
    // set the kinematic viscosity
    double nu_f = 1.0e-3;

    std::vector<dealii::Vector<double>> old_solution_values ( n_quad_spacetime , dealii::Vector<double> ( 3 ) );

    std::vector<dealii::Vector<double>> old_solution_dt ( n_quad_spacetime , dealii::Vector<double> ( 3 ) );

    std::vector < std::vector<dealii::Tensor<1,2> > >
        old_solution_grads ( n_quad_spacetime , std::vector<dealii::Tensor<1,2>> ( 3 ) );

    std::vector<dealii::Tensor<1,2>> solution_minus ( n_quad_space );

    std::vector<dealii::Tensor<1,2>> solution_plus ( n_quad_space );

    std::vector<dealii::Vector<double>> old_solution_plus ( n_quad_space , dealii::Vector<double> ( 3 ) );

    std::vector<dealii::Vector<double>> old_solution_minus ( n_quad_space , dealii::Vector<double> ( 3 ) );

    for ( const auto &cell_space : slab_its.dof->spatial ()->active_cell_iterators () )
    {
        // We only can calculate the contributions of processor local cells
        if ( cell_space->is_locally_owned () )
        {
            fe_values_spacetime.reinit_space ( cell_space );
            fe_jump_values_spacetime.reinit_space ( cell_space );

            cell_rhs = 0;
            for ( const auto &cell_time : slab_its.dof->temporal ()->active_cell_iterators () )
            {
                n = cell_time->index ();
                fe_values_spacetime.reinit_time ( cell_time );
                fe_jump_values_spacetime.reinit_time ( cell_time );
                fe_values_spacetime.get_local_dof_indices ( local_spacetime_dof_index );

                if ( n == 0 )
                {
                    fe_values_spacetime.spatial ()->get_function_values ( slab_initial_value ,
                                                                          old_solution_minus );
                }

                fe_values_spacetime.get_function_values( *slab_its.solution, old_solution_values );

                fe_values_spacetime.get_function_dt( *slab_its.solution , old_solution_dt );

                fe_values_spacetime.get_function_space_gradients( *slab_its.solution , old_solution_grads );

                for ( unsigned int q = 0 ; q < n_quad_spacetime ; ++q )
                {

                    dealii::Tensor < 1 , 2 > v;
                    dealii::Tensor < 1 , 2 > dt_v;
                    dealii::Tensor < 2 , 2 > grad_v;

                    const double p = old_solution_values[q] ( 2 );
                    for ( int c = 0 ; c < 2 ; c++ )
                    {
                        v[c] = old_solution_values[q]( c );
                        dt_v[c] = old_solution_dt[q]( c );
                        for ( int d = 0 ; d < 2 ; d++ )
                        {
                            grad_v[c][d] = old_solution_grads[q][c][d];
                        }
                    }
                    const double div_v = dealii::trace ( grad_v );

                    for ( unsigned int i = 0 ; i < dofs_per_spacetime_cell ; ++i )
                    {

                        // (dt u, v)
                        cell_rhs ( i+ n * dofs_per_spacetime_cell )
                           -= fe_values_spacetime.vector_value ( velocity , i , q )
                            * dt_v * fe_values_spacetime.JxW ( q );

                        // convection
                        cell_rhs ( i + n * dofs_per_spacetime_cell )
                           -= fe_values_spacetime.vector_value ( velocity , i , q )
                            * grad_v * v * fe_values_spacetime.JxW ( q );

                        // (grad u, grad v)
                        cell_rhs ( i + n * dofs_per_spacetime_cell )
                           -= nu_f
                           * dealii::scalar_product
                           (
                                   fe_values_spacetime.vector_space_grad ( velocity , i , q ) ,
                                   grad_v
                           )
                           * fe_values_spacetime.JxW ( q );

                        // (pressure gradient)
                        cell_rhs ( i + n * dofs_per_spacetime_cell )
                           += fe_values_spacetime.vector_divergence ( velocity , i , q )
                            * p * fe_values_spacetime.JxW ( q );

                        // (div free constraint)
                        cell_rhs ( i + n * dofs_per_spacetime_cell )
                           -= fe_values_spacetime.scalar_value ( pressure , i , q )
                            * div_v * fe_values_spacetime.JxW ( q );

                    } //dofs i
                } //quad

                fe_jump_values_spacetime.get_function_values_plus(*slab_its.solution , old_solution_plus);

                dealii::Tensor<1,2> v_plus;
                dealii::Tensor<1,2> v_minus;

                for ( unsigned int q = 0 ; q < n_quad_space ; ++q )
                {
                    for ( unsigned int c = 0 ; c < 2 ; ++c){
                        v_plus[c] = old_solution_plus[q](c);
                        v_minus[c] = old_solution_minus[q](c);
                    }

                    for ( unsigned int i = 0 ; i < dofs_per_spacetime_cell ; ++i )
                    {
                        // (v^+, u^+)
                        cell_rhs ( i + n * dofs_per_spacetime_cell )
                            -= fe_jump_values_spacetime.vector_value_plus ( velocity , i , q )
                             * v_plus * fe_jump_values_spacetime.JxW ( q );

                        // -(v^-, u^+)
                        cell_rhs ( i + n * dofs_per_spacetime_cell )
                            += fe_jump_values_spacetime.vector_value_plus ( velocity , i , q )
                             * v_minus * fe_jump_values_spacetime.JxW ( q );

                    } //dofs i
                } // quad_space
                if ( n < N - 1 )
                {
                    fe_jump_values_spacetime.get_function_values_minus(*slab_its.solution,old_solution_minus);
                }
            } //cell time
            slab_zero_constraints->distribute_local_to_global ( cell_rhs , local_spacetime_dof_index ,
                                                                slab_system_rhs );
        }
    } //cell space

    // We need to communicate local contributions to other processors after assembly
    slab_system_rhs.compress ( dealii::VectorOperation::add );
}

void Step4::solve_Newton_problem_on_slab ()
{
    pout << "Starting Newton solve" << std::endl;
    // The interface needs a solver control that is more or less irrelevant
    dealii::SolverControl sc ( 10000 , 1.0e-14 , false , false );
    // When Trilinos is compiled with MUMPS we prefer it.
    // If not installed switch to Amesos_Klu
    dealii::TrilinosWrappers::SolverDirect::AdditionalData ad ( false , "Amesos_Mumps" );
    auto solver = std::make_shared < dealii::TrilinosWrappers::SolverDirect > ( sc , ad );

    ////////////////////////////////////////////
    // Newton parameters
    ////////////////////////////////////////////
    double newton_lower_bound = 1.0e-10;
    unsigned int max_newton_steps = 10;
    unsigned int max_line_search_steps = 10;
    double newton_rebuild_parameter = 0.1;
    double newton_damping = 0.6;

    assemble_residual_on_slab ();
    double newton_residual = slab_system_rhs.linfty_norm ();
    double old_newton_residual;
    double new_newton_residual;

    unsigned int newton_step = 1;
    unsigned int line_search_step;

    pout << "0\t" << newton_residual << std::endl;
    while ( newton_residual > newton_lower_bound && newton_step <= max_newton_steps )
    {
        old_newton_residual = newton_residual;
        assemble_residual_on_slab ();
        newton_residual = slab_system_rhs.linfty_norm ();

        if ( newton_residual < newton_lower_bound )
        {
            pout << "res\t" << newton_residual << std::endl;
            break;
        }

        if ( newton_residual / old_newton_residual > newton_rebuild_parameter )
        {
            solver = nullptr;
            solver = std::make_shared < dealii::TrilinosWrappers::SolverDirect > ( sc , ad );
            assemble_system_on_slab ();

            solver->initialize ( slab_system_matrix );
        }

        solver->solve ( slab_newton_update , slab_system_rhs );
        slab_zero_constraints->distribute ( slab_newton_update );
        slab_owned_tmp = *slab_its.solution;
        for ( line_search_step = 0 ; line_search_step < max_line_search_steps ; line_search_step++ )
        {
            slab_owned_tmp += slab_newton_update;
            *slab_its.solution = slab_owned_tmp;
            assemble_residual_on_slab ();

            new_newton_residual = slab_system_rhs.linfty_norm ();
            if ( new_newton_residual < newton_residual )
                break;
            else
                slab_owned_tmp -= slab_newton_update;

            slab_newton_update *= newton_damping;
        }

        pout << std::setprecision ( 5 ) << newton_step << "\t" << std::scientific << newton_residual << "\t"
        << std::scientific
        << newton_residual / old_newton_residual << "\t";

        if ( newton_residual / old_newton_residual > newton_rebuild_parameter )
            pout << "r\t";
        else
            pout << " \t";

        pout << line_search_step << "\t" << std::scientific << std::endl;

        newton_step++;
    }

}

void Step4::output_results_on_slab ()
{
    auto n_dofs = slab_its.dof->n_dofs_time ();
    std::vector < std::string > field_names;
    std::vector < dealii::DataComponentInterpretation::DataComponentInterpretation > dci;
    for ( unsigned int i = 0 ; i < 2 ; i++ )
    {
        field_names.push_back ( "velocity" );
        dci.push_back ( dealii::DataComponentInterpretation::component_is_part_of_vector );
    }
    field_names.push_back ( "pressure" );
    dci.push_back ( dealii::DataComponentInterpretation::component_is_scalar );

    dealii::TrilinosWrappers::MPI::Vector tmp = *slab_its.solution;
    for ( unsigned i = 0 ; i < n_dofs ; i++ )
    {
        dealii::DataOut < 2 > data_out;
        data_out.attach_dof_handler ( *slab_its.dof->spatial () );
        dealii::TrilinosWrappers::MPI::Vector local_solution;
        local_solution.reinit ( space_locally_owned_dofs , space_locally_relevant_dofs , mpi_comm );

        idealii::slab::VectorTools::extract_subvector_at_time_dof ( tmp , local_solution , i );
        data_out.add_data_vector ( local_solution , field_names , dealii::DataOut < 2 > ::type_dof_data , dci );
        data_out.build_patches ( 2 );
        std::ostringstream filename;
        filename << "newton_navierstokes_solution_dG(" << fe.temporal ()->degree << ")_t_" << slab * n_dofs + i
        << ".vtu";
        // instead of a vtk we use the parallel write function
        data_out.write_vtu_in_parallel ( filename.str ().c_str () , mpi_comm );
    }
}
int main ( int argc , char *argv[] )
{
    //With MPI we need to begin with an InitFinalize call.
    dealii::Utilities::MPI::MPI_InitFinalize mpi ( argc , argv , MPIX_THREADS );
    // temporal finite element order
    // WARNING do not change as evaluation of temporal derivative of a given
    // space-time fe vector is not implemented yet.
    unsigned int r = 1;
    Step4 problem ( r );
    problem.run ();
}

