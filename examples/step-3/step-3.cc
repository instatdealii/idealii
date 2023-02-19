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

// number of thread parallel threads, not used so far
// but extension to dealii::WorkStream would be possible
#define MPIX_THREADS 1


////////////////////////////////////////////
// ideal.II includes
////////////////////////////////////////////

// now we use the parallel distributed instead of the sequential triangulation
#include <ideal.II/distributed/fixed_tria.hh>
// as well as trilinos linear algebra, so the collection is now over trilinos vectors
#include <ideal.II/lac/spacetime_trilinos_vector.hh>

//all other ideal.II includes are known
#include <ideal.II/base/time_iterator.hh>
#include <ideal.II/base/quadrature_lib.hh>
#include <ideal.II/dofs/spacetime_dof_handler.hh>
#include <ideal.II/dofs/slab_dof_tools.hh>
#include <ideal.II/fe/fe_dg.hh>
#include <ideal.II/fe/spacetime_fe_values.hh>
#include <ideal.II/numerics/vector_tools.hh>

////////////////////////////////////////////
// deal.II includes
////////////////////////////////////////////

// we only want process 0 to write to the console, ConditionalOStream provides this
#include <deal.II/base/conditional_ostream.h>


// now we use the parallel distributed instead of the sequential triangulation
#include <deal.II/distributed/tria.h>


#include <deal.II/base/function.h>

// and also distributed linear algebra provided by trilinos
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>


#include <deal.II/numerics/data_out.h>

////////////////////////////////////////////
// C++ includes
////////////////////////////////////////////
#include <fstream>

// right hand side function is (so far) the same as before
class RightHandSide: public dealii::Function<2>
{
public:
	virtual double value(const dealii::Point<2> &p,
			             const unsigned int component = 0);
};

double
RightHandSide::value(const dealii::Point<2> &p,
		             [[maybe_unused]] const unsigned int component){
	double t = get_time();
	const double x0 = 0.5+0.25*std::cos(2.*M_PI*t);
	const double x1 = 0.5+0.25*std::sin(2.*M_PI*t);
	double a = 50.;
	const double divisor = 1. + a*( (p[0]-x0)*(p[0]-x0) + (p[1]-x1)*(p[1]-x1) );
	double dtu =
		-( ( a * (p[0]-x0) * M_PI * std::sin(2.*M_PI*t) ) - ( a * (p[1]-x1) * M_PI * std::cos(2.*M_PI*t)) ) /
		(divisor*divisor);

	const double u_xx =
		-2.*a*( 1./(divisor*divisor)
				+2.*a*(p[0]-x0)*(p[0]-x0) * (-2./(divisor*divisor*divisor))
			    );

	const double u_yy =
		-2.*a*( 1./(divisor*divisor)
				+2.*a*(p[1]-x1)*(p[1]-x1) * (-2./(divisor*divisor*divisor))
				);

	return dtu -  (u_xx+u_yy);
}



class ExactSolution: public dealii::Function<2,double>
{
public:
	ExactSolution(): dealii::Function<2,double>(){}
	 double value(const dealii::Point<2> &p,
						 const unsigned int component = 0) const;
};

double ExactSolution::value(const dealii::Point<2> &p,
		[[maybe_unused]] const unsigned int component) const{
	const double t= get_time();
	const double x0 = 0.5+0.25*std::cos(2.*M_PI*t);
	const double x1 = 0.5+0.25*std::sin(2.*M_PI*t);
	return 1. / (
		1. + 50.*( (p[0]-x0)*(p[0]-x0) + (p[1]-x1)*(p[1]-x1) )
	);
}


class Step3
{
public:
	Step3(unsigned int spatial_degree,unsigned int temporal_degree);
	void run();

private:
	void make_grid();
	void time_marching();
	void setup_system_on_slab();
	void assemble_system_on_slab();
	void solve_system_on_slab();
	void output_results_on_slab();

	// we need to know the MPI communicator
	MPI_Comm mpi_comm;
	// Output based on some condition (here MPI process id = 0)
	dealii::ConditionalOStream pout;
    /////////////////////////////////////////////
	// space-time collections of slab objects
	/////////////////////////////////////////////
	// The triangulation is now parallel distributed
	idealii::spacetime::parallel::distributed::fixed::Triangulation<2> triangulation;
	idealii::spacetime::DoFHandler<2> dof_handler;
	idealii::spacetime::TrilinosVector solution;

	// The space-time finite element description
	idealii::spacetime::DG_FiniteElement<2> fe;

	////////////////////////////////////////////
	// objects needed on a single slab
	////////////////////////////////////////////
	dealii::SparsityPattern slab_sparsity_pattern;
	std::shared_ptr<dealii::AffineConstraints<double>> slab_constraints;
	dealii::TrilinosWrappers::SparseMatrix slab_system_matrix;
	dealii::TrilinosWrappers::MPI::Vector slab_system_rhs;
	dealii::TrilinosWrappers::MPI::Vector slab_initial_value;
	unsigned int slab;
	// For initial and Dirichlet boundary values it is simplest
	// to use the exact solution if known.
	ExactSolution exact_solution;

	// to allow for communication between locally owned and shared vectors
	// we sometimes need temporary vectors that we can write into
	dealii::TrilinosWrappers::MPI::Vector slab_owned_tmp;
	// set of space-time dofs owned by the processor
	dealii::IndexSet slab_locally_owned_dofs;
	// owned or belonging to owned elements (ghost dofs)
	dealii::IndexSet slab_locally_relevant_dofs;

	// same as above but just on the spatial grid
	dealii::IndexSet space_locally_owned_dofs;
	dealii::IndexSet space_locally_relevant_dofs;

	///////////////////////////////////////////////////////////////
	// Struct holding all iterators over the space-time objects.
	///////////////////////////////////////////////////////////////
	struct{
	  idealii::slab::parallel::distributed::TriaIterator<2> tria;
	  idealii::slab::DoFHandlerIterator<2> dof;
	  idealii::slab::TrilinosVectorIterator solution;
	} slab_its;
};


Step3::Step3(unsigned int spatial_degree,unsigned int temporal_degree):
				 // MPI communicator is set to world i.e. all nodes provided by MPI
				 mpi_comm(MPI_COMM_WORLD),
				 // Only output if process id is 0
				 pout(std::cout,dealii::Utilities::MPI::this_mpi_process(mpi_comm)==0),
				 triangulation(),
				 dof_handler(&triangulation),
				 fe(std::make_shared<dealii::FE_Q<2>>(spatial_degree),temporal_degree,
				    idealii::spacetime::DG_FiniteElement<2>::support_type::Legendre),
			     slab(0),
				 exact_solution(){}

void
Step3::run(){
	make_grid();
	time_marching();
}

void Step3::make_grid(){
	//construct an MPI parallel triangulation with the provided MPI communicator
	auto space_tria = std::make_shared<dealii::parallel::distributed::Triangulation<2>>(mpi_comm);
 	dealii::GridGenerator::hyper_cube(*space_tria);
 	const unsigned int M  = 5;
 	triangulation.generate(space_tria, M);
 	triangulation.refine_global(4,0);
 	dof_handler.generate();
}

void Step3::time_marching(){
	idealii::TimeIteratorCollection<2> tic = idealii::TimeIteratorCollection<2>();

	solution.reinit(triangulation.M());

	slab_its.tria = triangulation.begin();
	slab_its.dof = dof_handler.begin();
	slab_its.solution = solution.begin();
	slab = 0;
	tic.add_iterator(&slab_its.tria, &triangulation);
	tic.add_iterator(&slab_its.dof, &dof_handler);
	tic.add_iterator(&slab_its.solution, &solution);
	pout << "*******Starting time-stepping*********" << std::endl;
	for (; !tic.at_end() ; tic.increment()){
		pout << "Starting time-step ("
				  << slab_its.tria->startpoint() << ","
				  << slab_its.tria->endpoint() << "]" << std::endl;

		setup_system_on_slab();
		assemble_system_on_slab();
		solve_system_on_slab();
		output_results_on_slab();
		idealii::slab::VectorTools::extract_subvector_at_time_point(*slab_its.dof,
																	*slab_its.solution,
																	slab_initial_value,
																	slab_its.tria->endpoint()
																	);
		slab++;
	}
}

void Step3::setup_system_on_slab(){
	slab_its.dof->distribute_dofs(fe);
		pout << "Number of degrees of freedom: \n\t"
				  << slab_its.dof->n_dofs_space() << " (space) * "
				  << slab_its.dof->n_dofs_time() << " (time) = "
				  << slab_its.dof->n_dofs_spacetime() << std::endl;

	// we need to know the spatial set of degrees of freedom owned by the current MPI processor
	space_locally_owned_dofs = slab_its.dof->spatial()->locally_owned_dofs();
	// and beloging to elements owned by the processor
	dealii::DoFTools::extract_locally_relevant_dofs(*slab_its.dof->spatial(),space_locally_relevant_dofs);

	// The same holds for the set of space-time degrees of freedom
	slab_locally_owned_dofs = slab_its.dof->locally_owned_dofs();
	slab_locally_relevant_dofs =
			idealii::slab::DoFTools::extract_locally_relevant_dofs(*slab_its.dof);

	slab_owned_tmp.reinit(slab_locally_owned_dofs,mpi_comm);

	if ( slab == 0){
		// On the first slab the initial value has to be set to the correct
		// set of spatially relevant dofs
		slab_initial_value.reinit(space_locally_owned_dofs,space_locally_relevant_dofs,mpi_comm);

		// We only have full write access for locally owned vectors
		dealii::TrilinosWrappers::MPI::Vector tmp;
		tmp.reinit(space_locally_owned_dofs,mpi_comm);

		// set time of the exact solution to initial time point
		// (For a time independent initial value function this is not needed)
		exact_solution.set_time(0);
		// Interpolate the initial value into the FE space
		dealii::VectorTools::interpolate(
				*slab_its.dof->spatial(),
				exact_solution,
				tmp
		);
		// transfer locally owned initial value to locally relevant vector
		slab_initial_value = tmp;
	}

	//same as for the sequential case
	slab_constraints = std::make_shared<dealii::AffineConstraints<double>>();
	auto zero = dealii::Functions::ZeroFunction<2>();
	idealii::slab::VectorTools::interpolate_boundary_values(space_locally_relevant_dofs,
													  *slab_its.dof,
													  0,
													  zero,
													  slab_constraints);

	slab_constraints->close();

	dealii::DynamicSparsityPattern dsp(slab_its.dof->n_dofs_spacetime());
	idealii::slab::DoFTools::make_upwind_sparsity_pattern(*slab_its.dof,dsp);

	// To save memory we distribute the sparsity pattern to only hold the locally
	// needed set of space-time degrees of freedom
	dealii::SparsityTools::distribute_sparsity_pattern(
		dsp,
		slab_locally_owned_dofs,
		mpi_comm,
		slab_locally_relevant_dofs
	);

	// reinit the system matrix and vectors to hold only local information
	slab_system_matrix.reinit(slab_locally_owned_dofs,slab_locally_owned_dofs,dsp);

	slab_its.solution->reinit(slab_locally_owned_dofs,slab_locally_relevant_dofs,mpi_comm);
	slab_system_rhs.reinit(slab_locally_owned_dofs,mpi_comm);
}

void Step3::assemble_system_on_slab(){

	idealii::spacetime::QGauss<2> quad(fe.spatial()->degree+1,fe.temporal()->degree+1);


	idealii::spacetime::FEValues<2> fe_values_spacetime(fe,quad,
														dealii::update_values |
														dealii::update_gradients |
														dealii::update_quadrature_points |
														dealii::update_JxW_values);

	idealii::spacetime::FEJumpValues<2> fe_jump_values_spacetime(fe,quad,
																 dealii::update_values |
																 dealii::update_gradients |
																 dealii::update_quadrature_points |
																 dealii::update_JxW_values);

	RightHandSide right_hand_side;
	const unsigned int dofs_per_spacetime_cell = fe.dofs_per_cell;

	auto N = slab_its.tria->temporal()->n_global_active_cells();
	dealii::FullMatrix<double> cell_matrix(N*dofs_per_spacetime_cell,N*dofs_per_spacetime_cell);
	dealii::Vector<double> cell_rhs(N*dofs_per_spacetime_cell);

	std::vector<dealii::types::global_dof_index> local_spacetime_dof_index(N*dofs_per_spacetime_cell);

	unsigned int n;
	unsigned int n_quad_spacetime = fe_values_spacetime.n_quadrature_points;
	unsigned int n_quad_space = quad.spatial()->size();
	for (const auto &cell_space : slab_its.dof->spatial()->active_cell_iterators()){
	// We only can calculate the contributions of processor local cells
	// Otherwise the assembly is the same as in step-1
	if(cell_space ->is_locally_owned()){
		fe_values_spacetime.reinit_space(cell_space);
		fe_jump_values_spacetime.reinit_space(cell_space);
		std::vector<double> initial_values(fe_values_spacetime.spatial()->n_quadrature_points);
		fe_values_spacetime.spatial()->get_function_values(slab_initial_value, initial_values);

		cell_matrix = 0;
		cell_rhs = 0;
		for (const auto &cell_time : slab_its.dof->temporal()->active_cell_iterators()){
			n = cell_time->index();
			fe_values_spacetime.reinit_time(cell_time);
			fe_jump_values_spacetime.reinit_time(cell_time);
			fe_values_spacetime.get_local_dof_indices(local_spacetime_dof_index);

			for (unsigned int q = 0  ; q < n_quad_spacetime ; ++q){

				right_hand_side.set_time(fe_values_spacetime.time_quadrature_point(q));
				const auto &x_q = fe_values_spacetime.space_quadrature_point(q);
			for ( unsigned int i = 0 ; i < dofs_per_spacetime_cell ; ++i){
			for ( unsigned int j = 0 ; j < dofs_per_spacetime_cell ; ++j){

				// (dt u, v)
				cell_matrix(i + n*dofs_per_spacetime_cell,
						    j + n*dofs_per_spacetime_cell)
						+= fe_values_spacetime.shape_value(i,q) *
						   fe_values_spacetime.shape_dt(j,q) *
						   fe_values_spacetime.JxW(q);

				// (grad u, grad v)
				cell_matrix(i + n*dofs_per_spacetime_cell,
						    j + n*dofs_per_spacetime_cell)
						+= fe_values_spacetime.shape_space_grad(i,q) *
						   fe_values_spacetime.shape_space_grad(j,q) *
						   fe_values_spacetime.JxW(q);
			} //dofs j
				cell_rhs(i + n*dofs_per_spacetime_cell)
				+= fe_values_spacetime.shape_value(i,q) *
				   right_hand_side.value(x_q)*
				   fe_values_spacetime.JxW(q);
			} //dofs i
			} //quad

			//Jump terms and initial values just have a spatial loop
			for (unsigned int q = 0  ; q < n_quad_space ; ++q){
				for ( unsigned int i = 0 ; i < dofs_per_spacetime_cell ; ++i){
					for ( unsigned int j = 0 ; j < dofs_per_spacetime_cell ; ++j){
						// (v^+, u^+)
						cell_matrix(i + n*dofs_per_spacetime_cell,
									j + n*dofs_per_spacetime_cell)
								+= fe_jump_values_spacetime.shape_value_plus(i, q)*
								   fe_jump_values_spacetime.shape_value_plus(j, q)*
								   fe_jump_values_spacetime.JxW(q);

						// -(v^-, u^+)
						if(n>0){
							cell_matrix(i + n*dofs_per_spacetime_cell,
										j + (n-1)*dofs_per_spacetime_cell)
									-= fe_jump_values_spacetime.shape_value_plus(i, q)*
									   fe_jump_values_spacetime.shape_value_minus(j, q)*
									   fe_jump_values_spacetime.JxW(q);
						}
					} //dofs j
					if(n == 0){
						cell_rhs(i)
						  +=fe_jump_values_spacetime.shape_value_plus(i,q)*
						  initial_values[q]*
						  //value of previous solution at t0
						  fe_jump_values_spacetime.JxW(q);
					}
				}//dofs i
			}

	} //cell time
			slab_constraints->distribute_local_to_global(
					cell_matrix, cell_rhs, local_spacetime_dof_index,slab_system_matrix,slab_system_rhs);
	}} //cell space

	// We need to communicate local contributions to other processors after assembly
	slab_system_matrix.compress(dealii::VectorOperation::add);
	slab_system_rhs.compress(dealii::VectorOperation::add);
}

void Step3::solve_system_on_slab(){

	// The interface needs a solver control that is more or less irrelevant
	// as we use a direct solver without convergence criterion
	dealii::SolverControl sc(10000,1.0e-14,false,false);
	// When Trilinos is compiled with MUMPS we prefer it.
	// If not installed switch to Amesos_Klu
	dealii::TrilinosWrappers::SolverDirect::AdditionalData ad(false,"Amesos_Mumps");
	dealii::TrilinosWrappers::SolverDirect solver(sc,ad);
	// In this interface the factorize is called initialize
	solver.initialize(slab_system_matrix);
	// And vmult is called solve
	solver.solve(slab_owned_tmp, slab_system_rhs);
	slab_constraints->distribute(slab_owned_tmp);
	// The solver can only write into a vector with locally owned dofs
	// so we need to communicate between the processors
	*slab_its.solution = slab_owned_tmp;
}

void Step3::output_results_on_slab(){
	auto n_dofs = slab_its.dof->n_dofs_time();

	dealii::TrilinosWrappers::MPI::Vector tmp = *slab_its.solution;
	for ( unsigned i = 0 ; i < n_dofs ; i++){
		dealii::DataOut<2> data_out;
		data_out.attach_dof_handler(*slab_its.dof->spatial());
		dealii::TrilinosWrappers::MPI::Vector local_solution;
		local_solution.reinit(space_locally_owned_dofs,space_locally_relevant_dofs,mpi_comm);

		idealii::slab::VectorTools::extract_subvector_at_time_dof(tmp, local_solution,i);
		data_out.add_data_vector(local_solution,"Solution");
		data_out.build_patches();
		std::ostringstream filename;
		filename << "solution2_dG(" << fe.temporal()->degree
				<< ")_t_" << slab*n_dofs+i << ".vtu";
		// instead of a vtk we use the parallel write function
		data_out.write_vtu_in_parallel(filename.str().c_str(),mpi_comm);
	}
}
int main(int argc, char *argv[]) {
	//With MPI we need to begin with an InitFinalize call.
	dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, MPIX_THREADS);
	// spatial finite element order
	unsigned int s = 1;
	// temporal finite element order
	unsigned int r = 0;
	Step3 problem(s,r);
	problem.run();
}









