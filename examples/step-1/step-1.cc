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

////////////////////////////////////////////
//  Note:
//   To avoid mix-ups ideal.II header files
//   end on .hh instead of .h (deal.II)
//   apart from that, many includes are
//   called the same
////////////////////////////////////////////

// Include for the TimeIteratorCollection
#include <ideal.II/base/time_iterator.hh>
// Space-time quadrature formulae
#include <ideal.II/base/quadrature_lib.hh>

// Include DoF centered helper functions
#include <ideal.II/dofs/slab_dof_tools.hh>

// actual DoFHandler
#include <ideal.II/dofs/spacetime_dof_handler.hh>

// discontinuous Galerkin space-time elements
#include <ideal.II/fe/fe_dg.hh>
// FEValues for evaluation of tensor product shape functions
#include <ideal.II/fe/spacetime_fe_values.hh>

// Which triangulation to use, here: fixes spatial grid
#include <ideal.II/grid/fixed_tria.hh>

// Container class for standard deal vectors
#include <ideal.II/lac/spacetime_vector.hh>

// Include vector centered helper functions
#include <ideal.II/numerics/vector_tools.hh>

////////////////////////////////////////////
// deal.II includes
////////////////////////////////////////////

// needed for the spatial FE description
#include <deal.II/fe/fe_q.h>

// needed to generate underlying spatial grid
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

// All the linear algebra classes are used from deal.II directly
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

// ideal.II uses deal.II functions with set and get time
#include <deal.II/base/function.h>

// For now, output is done by hand
#include <deal.II/numerics/data_out.h>

////////////////////////////////////////////
// C++ includes
////////////////////////////////////////////

// needed for output into a file
#include <fstream>

/**
 * This function describes the exact solution.
 * It could be used to calculate L2-errors for example.
 * Here it is only used to give the full problem description
 */
class ExactSolution : public dealii::Function<2,double>
{
public:
    ExactSolution ()
    :
        dealii::Function<2,double> ()
        {
        }
    double
    value ( const dealii::Point<2> &p ,
            const unsigned int component = 0 ) const;
};

double ExactSolution::value (
        const dealii::Point<2> &p ,
        [[maybe_unused]] const unsigned int component ) const
{
    double t = get_time ();
    double x = p ( 0 );
    double y = p ( 1 );
    double return_value = ( x * x - x ) * ( y * y - y ) * t * 0.25;
    return return_value;
}

/**
 * This is the right hand side function of the heat equation.
 * It is derived by plugging in the exact solution above into the heat equation.
 *
 * Note the use of get_time(). This is how ideal.II handles time-dependent functions as this
 * functionality is already build into deal.II
 *
 */
class RightHandSide : public dealii::Function<2>
{
public:
    virtual double
    value ( const dealii::Point<2> &p , const unsigned int component = 0 );
};

double RightHandSide::value ( const dealii::Point<2> &p ,
                              [[maybe_unused]] const unsigned int component )
{
    double t = get_time ();
    double return_value = 0;
    double x = p ( 0 );
    double y = p ( 1 );
    return_value += ( x * x - x ) * 0.5 * t;
    return_value += ( y * y - y ) * 0.5 * t;
    return_value -= ( x * x - x ) * ( y * y - y ) * 0.25;
    return return_value;
}

// This class describes the solution of the heat equation with
// space-time slab tensor-product elements
class Step1
{
public:
    // We want to be able to easily switch spatial and temporal polynomial degree
    // i.e. bilinear/biquadratic elements in space and dG(0),dG(1),... elements in time.
    Step1 ( unsigned int spatial_degree , unsigned int temporal_degree );
    // This is the main function of the problem calling all other functions in the right order
    void
    run ();

private:
    // Construct a spatial mesh and use it to construct a resulting space-time mesh
    void
    make_grid ();
    // Iterate over all slabs in the space-time mesh
    void
    time_marching ();
    // Distribute degrees of freedom and reinitialize linear algebra objects to correct size.
    void
    setup_system_on_slab ();
    // Go over space-time elements in slab and calculate local contributions
    void
    assemble_system_on_slab ();
    // Solve the slab space-time linear equation system
    void
    solve_system_on_slab ();
    // Output the results of the current slab
    void
    output_results_on_slab ();

    /////////////////////////////////////////////
    // space-time collections of slab objects
    /////////////////////////////////////////////

    // The fixed triangulation shared a pointer to the spatial mesh
    // such that all slabs operate on the same mesh.
    // Without adaptivity this is the best choice as it saves on memory
    idealii::spacetime::fixed::Triangulation<2> triangulation;
    idealii::spacetime::DoFHandler<2> dof_handler;
    idealii::spacetime::Vector<double> solution;

    // The space-time finite element description
    idealii::spacetime::DG_FiniteElement<2> fe;

    ////////////////////////////////////////////
    // objects needed on a single slab
    ////////////////////////////////////////////
    dealii::SparsityPattern slab_sparsity_pattern;
    std::shared_ptr<dealii::AffineConstraints<double>> slab_constraints;
    dealii::SparseMatrix<double> slab_system_matrix;
    dealii::Vector<double> slab_system_rhs;
    dealii::Vector<double> slab_initial_value;
    unsigned int slab; // count the index of the current slab

    /////////////////////////////////////////////////////////////
    // Struct holding all iterators over the space-time objects.
    //   this is completely optional but hopefully enhances
    //   readability as it becomes clear that these are iterators.
    /////////////////////////////////////////////////////////////
    struct
    {
        idealii::slab::TriaIterator<2> tria;
        idealii::slab::DoFHandlerIterator<2> dof;
        idealii::slab::VectorIterator<double> solution;
    } slab_its;

};

// constructor
Step1::Step1 ( unsigned int spatial_degree , unsigned int temporal_degree )
:
                // space-time triangulation
                triangulation (),
                // DoFHandler has to get pointer to space-time tria
                // as different types (e.g. MPI parallel triangulations) can't be derived
                // from a common base class due to lists of shared pointers.
                dof_handler ( &triangulation ),
                // space-time finite element with
                // * continuous Lagrangian FE_Q finite element in space of order spatial_degree
                // * discontinuous Lagrangian FE_DGQ finite element in time of order temporal_degree
                fe ( std::make_shared < dealii::FE_Q < 2 >> ( spatial_degree ) ,
                     temporal_degree ),
                     // initialize current slab number to 0
                     slab ( 0 )
{
}

// This functions is much shorter compared to the stationary versions
// as the common FE-code loop is inside the time_marching function
void Step1::run ()
{
    make_grid ();
    time_marching ();
}

void Step1::make_grid ()
{
    // construct a shared pointer to a spatial triangulation
    auto space_tria = std::make_shared<dealii::Triangulation<2>> ();
    // generate a unit square spatial domain
    dealii::GridGenerator::hyper_cube ( *space_tria );
    // number of initial slabs in the triangulation
    const unsigned int M = 5;
    // Fill the internal list with M slab::Triangulation objects sharing the same spatial triangulation
    triangulation.generate ( space_tria , M );
    // Refine the grids on each slab in (space,time)
    triangulation.refine_global ( 4 , 0 );
    // Generate a slab::DoFHandler for each slab::Triangulation
    dof_handler.generate ();
}

void Step1::time_marching ()
{
    // This collection simplifies time marching and increments all
    // registered spacetime iterators at the same time
    idealii::TimeIteratorCollection < 2 > tic = idealii::TimeIteratorCollection<
            2> ();

    // Fill the internal list of vectors with M dealii::Vector<double> vectors.
    solution.reinit ( triangulation.M () );

    // Get iterators to the first slabs
    slab_its.tria = triangulation.begin ();
    slab_its.dof = dof_handler.begin ();
    slab_its.solution = solution.begin ();

    // Register iterators with the TimeIteratorCollection
    tic.add_iterator ( &slab_its.tria , &triangulation );
    tic.add_iterator ( &slab_its.dof , &dof_handler );
    tic.add_iterator ( &slab_its.solution , &solution );
    std::cout << "*******Starting time-stepping*********" << std::endl;

    slab = 0;
    // Iterate over the iterator collection
    for ( ; !tic.at_end () ; tic.increment () )
    {
        std::cout << "Starting time-step (" << slab_its.tria->startpoint ()
                << "," << slab_its.tria->endpoint () << "]" << std::endl;

        // this is the "typical" FE-code loop without the make_grid function
        setup_system_on_slab ();
        assemble_system_on_slab ();
        solve_system_on_slab ();
        output_results_on_slab ();
        // Take the slab solution subvector at the final temporal dof of the current slab
        // This is needed as the initial value for the next slab
        idealii::slab::VectorTools::extract_subvector_at_time_dof (
                *slab_its.solution , slab_initial_value ,
                slab_its.tria->temporal ()->n_global_active_cells () - 1 );
        slab++;
    }
}

void Step1::setup_system_on_slab ()
{

    // Distribute spatial and temporal dofs
    slab_its.dof->distribute_dofs ( fe );
    std::cout << "Number of degrees of freedom: \n\t"
            << slab_its.dof->n_dofs_space () << " (space) * "
            << slab_its.dof->n_dofs_time () << " (time) = "
            << slab_its.dof->n_dofs_spacetime () << std::endl;

    // On the first slab the initial value vector has to be set to the correct (spatial) size.
    if ( slab == 0 )
    {
        slab_initial_value.reinit ( slab_its.dof->n_dofs_space () );
        // For the above ExactSolution the initial value is 0
        slab_initial_value = 0;
    }

    // construct the constraint object
    slab_constraints = std::make_shared<dealii::AffineConstraints<double>> ();

    // The Dirichlet boundary condition for the ExactSolution is homogeneous
    auto zero = dealii::Functions::ZeroFunction<2> ();
    // Add Dirichlet boundary values to the constraint object.
    idealii::slab::VectorTools::interpolate_boundary_values (
            *slab_its.dof , 0 , zero , slab_constraints );

    slab_constraints->close ();

    // initialize a dynamic sparsity pattern to the slab size
    dealii::DynamicSparsityPattern dsp ( slab_its.dof->n_dofs_spacetime () );
    // construct the space-time sparsity pattern
    // Upwind means we solve forward in time and need information for
    // internal jump terms for slabs with more than one temporal element.
    idealii::slab::DoFTools::make_upwind_sparsity_pattern ( *slab_its.dof ,
                                                            dsp );

    // as in deal.II
    slab_sparsity_pattern.copy_from ( dsp );
    slab_system_matrix.reinit ( slab_sparsity_pattern );

    // set solution and right hand side to the correct number of unknowns
    slab_its.solution->reinit ( slab_its.dof->n_dofs_spacetime () );
    slab_system_rhs.reinit ( slab_its.dof->n_dofs_spacetime () );
}

void Step1::assemble_system_on_slab ()
{

    // Similar to the stationary case we start with a quadrature and FEValues objects
    idealii::spacetime::QGauss < 2 > quad ( fe.spatial ()->degree + 1 ,
                                            fe.temporal ()->degree + 1 );

    idealii::spacetime::FEValues < 2 > fe_values_spacetime (
            fe ,
            quad ,
            dealii::update_values | dealii::update_gradients
            | dealii::update_quadrature_points | dealii::update_JxW_values );

    // To account for jump values we use the FEJumpValues object which is somewhat similar
    // to FEFaceValues objects.
    idealii::spacetime::FEJumpValues < 2 > fe_jump_values_spacetime (
            fe ,
            quad ,
            dealii::update_values | dealii::update_gradients
            | dealii::update_quadrature_points | dealii::update_JxW_values );

    RightHandSide right_hand_side;

    // Get the number of dofs in a space-time elmement
    const unsigned int dofs_per_spacetime_cell = fe.dofs_per_cell;

    // Number of temporal elements
    unsigned int N = slab_its.tria->temporal ()->n_global_active_cells ();

    // The easiest way to account for jump values is to construct local objects
    // extruded in the temporal direction, such that they hold information over all
    // temporal elements but just the information of a single spatial element
    dealii::FullMatrix<double> cell_matrix ( N * dofs_per_spacetime_cell ,
                                             N * dofs_per_spacetime_cell );
    dealii::Vector<double> cell_rhs ( N * dofs_per_spacetime_cell );

    std::vector < dealii::types::global_dof_index > local_spacetime_dof_index (
            N * dofs_per_spacetime_cell );

    // index of the current temporal element to calculate index offset
    unsigned int n;
    // number of space-time and space quadrature points
    unsigned int n_quad_spacetime = fe_values_spacetime.n_quadrature_points;
    unsigned int n_quad_space = quad.spatial ()->size ();

    // First, iterate over active spatial elements
    for ( const auto &cell_space : slab_its.dof->spatial ()->active_cell_iterators () )
    {
        // recalculate local information for the current spatial element
        fe_values_spacetime.reinit_space ( cell_space );
        fe_jump_values_spacetime.reinit_space ( cell_space );

        // get local contribution of the slab_initial_value vector
        std::vector<double> initial_values (
                fe_values_spacetime.spatial ()->n_quadrature_points );
        fe_values_spacetime.spatial ()->get_function_values (
                slab_initial_value , initial_values );

        // Reset local contributions to 0
        cell_matrix = 0;
        cell_rhs = 0;

        // Second, iterator over active temporal elements
        for ( const auto &cell_time : slab_its.dof->temporal ()->active_cell_iterators () )
        {
            // As the temporal elements are part of a 1-d triangulation
            // the indices are ordered from left to right, i.e. startpoint to endpoint
            n = cell_time->index ();

            // recalculate local information for the current temporal element
            fe_values_spacetime.reinit_time ( cell_time );
            fe_jump_values_spacetime.reinit_time ( cell_time );
            // get local space-time dof indices for the current
            // cell_space x cell_time tensor product cell
            fe_values_spacetime.get_local_dof_indices (
                    local_spacetime_dof_index );

            // Iterate over space-time quadrature points for
            // all contributions except for jump terms
            for ( unsigned int q = 0 ; q < n_quad_spacetime ; ++q )
            {

                // set the time of the right hand side to the time current quadrature point
                right_hand_side.set_time (
                        fe_values_spacetime.time_quadrature_point ( q ) );
                // get the spatial location of the current quadrature point
                const auto &x_q = fe_values_spacetime.space_quadrature_point (
                        q );
                // iterate over all space-time dofs of the current element
                // and calculate contributions at current quadrature point
                for ( unsigned int i = 0 ; i < dofs_per_spacetime_cell ; ++i )
                {
                    for ( unsigned int j = 0 ; j < dofs_per_spacetime_cell ;
                            ++j )
                    {

                        // Note that for unsymmetric matrices the indices are flipped
                        // in comparison to the usual weak form derivation by multiplying
                        // test functions from the right.

                        // offset indices by the past space-time dofs of previous
                        // temporal elements

                        // (dt u, v)
                        cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                      j + n * dofs_per_spacetime_cell ) +=
                                              fe_values_spacetime.shape_value ( i , q ) * fe_values_spacetime.shape_dt (
                                                      j , q )
                                                      * fe_values_spacetime.JxW ( q );

                        // (grad u, grad v)
                        cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                      j + n * dofs_per_spacetime_cell ) +=
                                              fe_values_spacetime.shape_space_grad ( i , q ) * fe_values_spacetime.shape_space_grad (
                                                      j , q )
                                                      * fe_values_spacetime.JxW ( q );
                    } //dofs j
                    // Calculate local contribution of the right hand side function.
                    // Note that the correct time was set above so it still is an evaluation
                    // f(t,x)
                    cell_rhs ( i + n * dofs_per_spacetime_cell ) +=
                            fe_values_spacetime.shape_value ( i , q ) * right_hand_side.value (
                                    x_q )
                                    * fe_values_spacetime.JxW ( q );
                } //dofs i
            } //quad

            //Jump terms and initial values just have a spatial quadrature loop
            for ( unsigned int q = 0 ; q < n_quad_space ; ++q )
            {
                // However, dof loops are still over space-time indices
                // as different temporal support points might not have a
                // degree of freedom at the edges of the temporal element
                // and so all shape functions contribute to the jump values
                for ( unsigned int i = 0 ; i < dofs_per_spacetime_cell ; ++i )
                {
                    for ( unsigned int j = 0 ; j < dofs_per_spacetime_cell ;
                            ++j )
                    {
                        // (v^+, u^+)
                        cell_matrix ( i + n * dofs_per_spacetime_cell ,
                                      j + n * dofs_per_spacetime_cell ) +=
                                              fe_jump_values_spacetime.shape_value_plus ( i ,
                                                                                          q )
                                                                                          * fe_jump_values_spacetime.shape_value_plus (
                                                                                                  j , q )
                                                                                                  * fe_jump_values_spacetime.JxW ( q );

                        // -(v^+, u^-)
                        // If we have more than a single element per slab
                        // We have to account for the inner jumps between elements
                        // Therefore the column index is offset by one temporal element
                        if ( n > 0 )
                        {
                            cell_matrix (
                                    i + n * dofs_per_spacetime_cell ,
                                    j + ( n - 1 ) * dofs_per_spacetime_cell ) -=
                                            fe_jump_values_spacetime.shape_value_plus (
                                                    i , q )
                                                    * fe_jump_values_spacetime.shape_value_minus (
                                                            j , q )
                                                            * fe_jump_values_spacetime.JxW ( q );
                        }
                    } //dofs j
                    // The first temporal element has to account for the jump to the previous
                    // slab or the initial value in the right hand side.
                    if ( n == 0 )
                    {
                        cell_rhs ( i ) +=
                                fe_jump_values_spacetime.shape_value_plus ( i ,
                                                                            q )
                                                                            * initial_values[q] *
                                                                            //value of previous solution at t0
                                                                            fe_jump_values_spacetime.JxW ( q );
                    }
                }					//dofs i
            }

        } //cell time
        // after contributions for all temporal elements are calculated,
        // write this information into the slab system matrix and rhs
        slab_constraints->distribute_local_to_global (
                cell_matrix , cell_rhs , local_spacetime_dof_index ,
                slab_system_matrix , slab_system_rhs );
    } //cell space
}

void Step1::solve_system_on_slab ()
{
    // Simply use a direct solver here. CG would not work as the matrix is not symmetric.
    dealii::SparseDirectUMFPACK solver;
    solver.factorize ( slab_system_matrix );
    solver.vmult ( *slab_its.solution , slab_system_rhs );
    // after solving set the correct Dirichlet values in the solution
    slab_constraints->distribute ( *slab_its.solution );
}

// For now the output is done somewhat by hand.
// But a spacetime or slab DataOut object is planned/WIP.
void Step1::output_results_on_slab ()
{
    // Simply output the subvectors at all temporal degrees of freedom.
    auto n_dofs = slab_its.dof->n_dofs_time ();
    for ( unsigned i = 0 ; i < n_dofs ; i++ )
    {
        // Construct a spatial DataOut object
        dealii::DataOut < 2 > data_out;
        // Register the spatial dof handler
        data_out.attach_dof_handler ( *slab_its.dof->spatial () );

        // extract the spatial solution at the current temporal dof
        dealii::Vector<double> local_solution;
        local_solution.reinit ( slab_its.dof->n_dofs_space () );
        idealii::slab::VectorTools::extract_subvector_at_time_dof (
                *slab_its.solution , local_solution , i );

        // add this vector to the DataOut object
        data_out.add_data_vector ( local_solution , "Solution" );
        data_out.build_patches ();

        // construct a filename, for a fixed triangulation with uniform refinement
        // this is relatively easy as all slabs have the same number of temporal degrees of freedom.
        // Later slabs are just offset
        std::ostringstream filename;
        filename << "solution_split_dG(" << fe.temporal ()->degree << ")_t_"
                << slab * n_dofs + i << ".vtk";

        std::ofstream output ( filename.str () );
        data_out.write_vtk ( output );
        output.close ();
    }
}
int main ()
{
    // spatial finite element order
    unsigned int s = 1;
    // temporal finite element order
    unsigned int r = 1;
    // Run the problem with cG(s)dG(r) finite elements.
    Step1 problem ( s , r );
    problem.run ();
}

