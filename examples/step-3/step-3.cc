//*---------------------------------------------------------------------
//
// Copyright (C) 2022 - 2024 by the ideal.II authors
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
//*---------------------------------------------------------------------

//////////////////////////////////////////
// @<H2> include files
//////////////////////////////////////////

////////////////////////////////////////////
// @<H3> ideal.II includes
////////////////////////////////////////////

// Here, we use the parallel distributed triangulation and linear algebra
// (provided by Trilinos) so we have to use the matching includes.
#include <ideal.II/distributed/fixed_tria.hh>

#include <ideal.II/lac/spacetime_trilinos_vector.hh>

#include <cmath>

// All other ideal.II includes are known
#include <ideal.II/base/quadrature_lib.hh>
#include <ideal.II/base/time_iterator.hh>

#include <ideal.II/dofs/slab_dof_tools.hh>
#include <ideal.II/dofs/spacetime_dof_handler.hh>

#include <ideal.II/fe/fe_dg.hh>
#include <ideal.II/fe/spacetime_fe_values.hh>

#include <ideal.II/numerics/vector_tools.hh>
// As we need the support type quite often we shorten this lengthy typename with
// an alias
using DGFE = idealii::spacetime::DG_FiniteElement<2>;

////////////////////////////////////////////
// @<H3> deal.II includes
////////////////////////////////////////////

// Most includes are the same as before except for the
// distributed versions of triangulation and linear algebra objects.
// Additionally, we include the conditional output stream to suppress
// duplicate output on other processors.
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

////////////////////////////////////////////
// @<H3> Trilinos and C++ includes
////////////////////////////////////////////
// To simplify handling of command line arguments
// in ``main()``
#include <Teuchos_CommandLineProcessor.hpp>

#include <fstream>

////////////////////////////////////////////
// @<H2> Space-time functions
////////////////////////////////////////////
//
// The manufactured exact solution :math:`u` is described in [Hartmann1998]_.
// The right hand side function is derived by inserting :math:`u` into the heat
// equation.

class ExactSolution : public dealii::Function<2, double>
{
public:
  ExactSolution()
    : dealii::Function<2, double>()
  {}
  double
  value(const dealii::Point<2> &p, const unsigned int component = 0) const;
};

double
ExactSolution::value(const dealii::Point<2>             &p,
                     [[maybe_unused]] const unsigned int component) const
{
  const double t  = get_time();
  const double x0 = 0.5 + 0.25 * std::cos(2. * M_PI * t);
  const double x1 = 0.5 + 0.25 * std::sin(2. * M_PI * t);
  return 1. /
         (1. + 50. * ((p[0] - x0) * (p[0] - x0) + (p[1] - x1) * (p[1] - x1)));
}

class RightHandSide : public dealii::Function<2>
{
public:
  virtual double
  value(const dealii::Point<2> &p, const unsigned int component = 0);
};

double
RightHandSide::value(const dealii::Point<2>             &p,
                     [[maybe_unused]] const unsigned int component)
{
  double       t  = get_time();
  const double x0 = 0.5 + 0.25 * std::cos(2. * M_PI * t);
  const double x1 = 0.5 + 0.25 * std::sin(2. * M_PI * t);
  double       a  = 50.;
  const double divisor =
    1. + a * ((p[0] - x0) * (p[0] - x0) + (p[1] - x1) * (p[1] - x1));

  double dtu = -((a * (p[0] - x0) * M_PI * std::sin(2. * M_PI * t)) -
                 (a * (p[1] - x1) * M_PI * std::cos(2. * M_PI * t))) /
               (divisor * divisor);

  const double u_xx =
    -2. * a *
    (1. / (divisor * divisor) + 2. * a * (p[0] - x0) * (p[0] - x0) *
                                  (-2. / (divisor * divisor * divisor)));

  const double u_yy =
    -2. * a *
    (1. / (divisor * divisor) + 2. * a * (p[1] - x1) * (p[1] - x1) *
                                  (-2. / (divisor * divisor * divisor)));

  return dtu - (u_xx + u_yy);
}

////////////////////////////////////////////
// @<H2> The Step3 class
////////////////////////////////////////////
//
class Step3
{
public:
  Step3(unsigned int       spatial_degree,
        unsigned int       temporal_degree,
        unsigned int       M,
        unsigned int       n_ref_space,
        DGFE::support_type support_type,
        bool               write_vtu);
  void
  run();

private:
  void
  make_grid();
  void
  time_marching();
  void
  setup_system_on_slab();
  void
  assemble_system_on_slab();
  void
  solve_system_on_slab();
  void
  process_results_on_slab();
  void
  output_results_on_slab();

  MPI_Comm     mpi_comm;    // MPI Communicator
  unsigned int M;           // Number of temporal elements
  unsigned int n_ref_space; // number of spatial refinements
  bool         write_vtu;   // output results?

  dealii::ConditionalOStream pout; // To allow output only on MPI rank 0
  // The triangulation is now parallel distributed and the matrices and vectors
  // are provided by Trilinos.
  idealii::spacetime::parallel::distributed::fixed::Triangulation<2>
                                     triangulation;
  idealii::spacetime::DoFHandler<2>  dof_handler;
  idealii::spacetime::TrilinosVector solution;

  idealii::spacetime::DG_FiniteElement<2> fe;

  dealii::SparsityPattern                            slab_sparsity_pattern;
  std::shared_ptr<dealii::AffineConstraints<double>> slab_constraints;
  dealii::TrilinosWrappers::SparseMatrix             slab_system_matrix;
  dealii::TrilinosWrappers::MPI::Vector              slab_system_rhs;
  dealii::TrilinosWrappers::MPI::Vector              slab_initial_value;
  unsigned int                                       slab;
  ExactSolution                                      exact_solution;
  double                                             L2_sqr_error;
  unsigned int                                       dofs_total;

  // For parallel communication we need some additional information.
  // First we need to know which DoF indices are owned by the current MPI rank.
  // Then, we need to know which indices are relevant, i.e. owned or ghost
  // entries needed for assembly but belonging to a different rank. We need
  // those both for the complete slab and for the spatial grid in different
  // points of the code. Finally, we sometimes need temporary vectors that the
  // current rank is allowed to write into for communication between different
  // ranks.
  dealii::IndexSet                      slab_locally_owned_dofs;
  dealii::IndexSet                      slab_locally_relevant_dofs;
  dealii::IndexSet                      space_locally_owned_dofs;
  dealii::IndexSet                      space_locally_relevant_dofs;
  dealii::TrilinosWrappers::MPI::Vector slab_owned_tmp;

  struct
  {
    idealii::slab::parallel::distributed::TriaIterator<2> tria;
    idealii::slab::DoFHandlerIterator<2>                  dof;
    idealii::slab::TrilinosVectorIterator                 solution;
  } slab_its;
};
////////////////////////////////////////////
// @<H3> Step3::Step3
////////////////////////////////////////////
// The constructor is almost the same as for step-1 as we are solving the same
// PDE. The main differences are
//
// * A larger set of parameters to support a parameter study,
// * Initializing the MPI communicator. Here to world, i.e. all ranks,
// * Restricting conditional output to rank 0,
// * Allowing the temporal support type to change.
//
Step3::Step3(unsigned int       spatial_degree,
             unsigned int       temporal_degree,
             unsigned int       M,
             unsigned int       n_ref_space,
             DGFE::support_type support_type,
             bool               write_vtu)
  : mpi_comm(MPI_COMM_WORLD)
  , pout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  , triangulation()
  , dof_handler(&triangulation)
  , fe(std::make_shared<dealii::FE_Q<2>>(spatial_degree),
       temporal_degree,
       support_type)
  , M(M)
  , n_ref_space(n_ref_space)
  , write_vtu(write_vtu)
  , slab(0)
  , exact_solution()
  , L2_sqr_error(0.)
  , dofs_total(0)
{}

////////////////////////////////////////////
// @<H3> Step3::run
////////////////////////////////////////////
//
void
Step3::run() // same as before
{
  make_grid();
  time_marching();
}

////////////////////////////////////////////
// @<H3> Step3::make_grid
////////////////////////////////////////////
//
void
Step3::make_grid()
{
  auto space_tria = // Construct an MPI parallel triangulation
    std::make_shared<dealii::parallel::distributed::Triangulation<2>>(mpi_comm);
  dealii::GridGenerator::hyper_cube(*space_tria);
  triangulation.generate(space_tria, M);
  triangulation.refine_global(n_ref_space, 0);
  dof_handler.generate();
}

////////////////////////////////////////////
// @<H3> Step3::time_marching
////////////////////////////////////////////
// We now want to assess the performance of the different discretizations
// so we will calculate local contributions to the :math:`L^2` error in
// ``process_results_on_slab()``.
// Additionally, we only do output if ``write_vtu`` is true.

void
Step3::time_marching()
{
  idealii::TimeIteratorCollection<2> tic = idealii::TimeIteratorCollection<2>();

  solution.reinit(triangulation.M());

  slab_its.tria     = triangulation.begin();
  slab_its.dof      = dof_handler.begin();
  slab_its.solution = solution.begin();
  slab              = 0;
  tic.add_iterator(&slab_its.tria, &triangulation);
  tic.add_iterator(&slab_its.dof, &dof_handler);
  tic.add_iterator(&slab_its.solution, &solution);
  pout << "*******Starting time-stepping*********" << std::endl;
  for (; !tic.at_end(); tic.increment())
    {
      pout << "Starting time-step (" << slab_its.tria->startpoint() << ","
           << slab_its.tria->endpoint() << "]" << std::endl;

      setup_system_on_slab();
      assemble_system_on_slab();
      solve_system_on_slab();
      process_results_on_slab();
      if (write_vtu)
        output_results_on_slab();
      // For some support points the final DoF of the slab is no longer at the
      // final time point. Therefore, we have to calculate the value as a linear
      // combination of the DoF values at the final temporal element.
      // This is done by the following extract function.
      idealii::slab::VectorTools::extract_subvector_at_time_point(
        *slab_its.dof,
        *slab_its.solution,
        slab_initial_value,
        slab_its.tria->endpoint());
      slab++;
    }
  double L2_error = std::sqrt(L2_sqr_error);
  pout << "Total number of space-time DoFs:\n\t" << dofs_total << std::endl;
  pout << "L2_error:\n\t" << L2_error << std::endl;
}

////////////////////////////////////////////
// @<H3> Step3::setup_system_on_slab
////////////////////////////////////////////
//
void
Step3::setup_system_on_slab()
{
  slab_its.dof->distribute_dofs(fe);
  pout << "Number of degrees of freedom: \n\t" << slab_its.dof->n_dofs_space()
       << " (space) * " << slab_its.dof->n_dofs_time()
       << " (time) = " << slab_its.dof->n_dofs_spacetime() << std::endl;

  // Tally the total amount of space-time
  dofs_total += slab_its.dof->n_dofs_spacetime();
  // Extract the spatial DoF indices by accessing the ``spatial()``
  // component of the ``slab::DoFHandler``.
  space_locally_owned_dofs = slab_its.dof->spatial()->locally_owned_dofs();
  dealii::DoFTools::extract_locally_relevant_dofs(*slab_its.dof->spatial(),
                                                  space_locally_relevant_dofs);

  // The slab DoF indices are extracted in a similar way, but by using functions
  // provided in ideal.II.
  slab_locally_owned_dofs = slab_its.dof->locally_owned_dofs();
  slab_locally_relevant_dofs =
    idealii::slab::DoFTools::extract_locally_relevant_dofs(*slab_its.dof);

  slab_owned_tmp.reinit(slab_locally_owned_dofs, mpi_comm);

  if (slab == 0)
    {
      // On the first slab the initial value has to be set to the set of
      // spatially relevant dofs
      slab_initial_value.reinit(space_locally_owned_dofs,
                                space_locally_relevant_dofs,
                                mpi_comm);

      // We only have write access to locally owned vectors.
      // Compared to ``slab_owned_tmp`` we need this temporary vector only on
      // the first slab for interpolation of the initial value into the FE
      // space.
      dealii::TrilinosWrappers::MPI::Vector tmp;
      tmp.reinit(space_locally_owned_dofs, mpi_comm);

      // Set time of the exact solution to initial time point
      // (For a time independent initial value function this is not necessary)
      exact_solution.set_time(0);
      // Interpolate the initial value into the FE space
      dealii::VectorTools::interpolate(*slab_its.dof->spatial(),
                                       exact_solution,
                                       tmp);
      // Communicate locally owned initial value to locally relevant vector
      slab_initial_value = tmp;
    }

  // For the interpolation of boundary values we pass the relevant index set.
  slab_constraints = std::make_shared<dealii::AffineConstraints<double>>();
  idealii::slab::VectorTools::interpolate_boundary_values(
    space_locally_relevant_dofs,
    *slab_its.dof,
    0,
    exact_solution,
    slab_constraints);
  slab_constraints->close();

  dealii::DynamicSparsityPattern dsp(slab_its.dof->n_dofs_spacetime());
  idealii::slab::DoFTools::make_upwind_sparsity_pattern(*slab_its.dof, dsp);

  // To save memory we distribute the sparsity pattern to only hold the locally
  // needed set of space-time degrees of freedom.
  dealii::SparsityTools::distribute_sparsity_pattern(
    dsp, slab_locally_owned_dofs, mpi_comm, slab_locally_relevant_dofs);

  slab_system_matrix.reinit(slab_locally_owned_dofs,
                            slab_locally_owned_dofs,
                            dsp);

  slab_its.solution->reinit(slab_locally_owned_dofs,
                            slab_locally_relevant_dofs,
                            mpi_comm);
  slab_system_rhs.reinit(slab_locally_owned_dofs, mpi_comm);
}

////////////////////////////////////////////
// @<H3> Step3::assemble_system_on_slab
////////////////////////////////////////////
// We only can calculate the contributions of processor local cells.
// Otherwise the assembly is the same as in step-1
void
Step3::assemble_system_on_slab()
{
  idealii::spacetime::QGauss<2> quad(fe.spatial()->degree + 2,
                                     fe.temporal()->degree + 2);

  idealii::spacetime::FEValues<2> fe_values_spacetime(
    fe,
    quad,
    dealii::update_values | dealii::update_gradients |
      dealii::update_quadrature_points | dealii::update_JxW_values);

  idealii::spacetime::FEJumpValues<2> fe_jump_values_spacetime(
    fe,
    quad,
    dealii::update_values | dealii::update_gradients |
      dealii::update_quadrature_points | dealii::update_JxW_values);

  RightHandSide      right_hand_side;
  const unsigned int dofs_per_spacetime_cell = fe.dofs_per_cell;

  auto N = slab_its.tria->temporal()->n_global_active_cells();
  dealii::FullMatrix<double> cell_matrix(N * dofs_per_spacetime_cell,
                                         N * dofs_per_spacetime_cell);
  dealii::Vector<double>     cell_rhs(N * dofs_per_spacetime_cell);

  std::vector<dealii::types::global_dof_index> local_spacetime_dof_index(
    N * dofs_per_spacetime_cell);

  unsigned int n;
  unsigned int n_quad_spacetime = fe_values_spacetime.n_quadrature_points;
  unsigned int n_quad_space     = quad.spatial()->size();
  for (const auto &cell_space :
       slab_its.dof->spatial()->active_cell_iterators())
    {
      if (cell_space->is_locally_owned())
        {
          fe_values_spacetime.reinit_space(cell_space);
          fe_jump_values_spacetime.reinit_space(cell_space);
          std::vector<double> initial_values(
            fe_values_spacetime.spatial()->n_quadrature_points);
          fe_values_spacetime.spatial()->get_function_values(slab_initial_value,
                                                             initial_values);

          cell_matrix = 0;
          cell_rhs    = 0;
          for (const auto &cell_time :
               slab_its.dof->temporal()->active_cell_iterators())
            {
              n = cell_time->index();
              fe_values_spacetime.reinit_time(cell_time);
              fe_jump_values_spacetime.reinit_time(cell_time);
              fe_values_spacetime.get_local_dof_indices(
                local_spacetime_dof_index);

              for (unsigned int q = 0; q < n_quad_spacetime; ++q)
                {
                  right_hand_side.set_time(
                    fe_values_spacetime.time_quadrature_point(q));
                  const auto &x_q =
                    fe_values_spacetime.space_quadrature_point(q);
                  for (unsigned int i = 0; i < dofs_per_spacetime_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_spacetime_cell; ++j)
                        {
                          // :math:`(\partial_t u,v)`
                          cell_matrix(i + n * dofs_per_spacetime_cell,
                                      j + n * dofs_per_spacetime_cell) +=
                            fe_values_spacetime.shape_value(i, q) *
                            fe_values_spacetime.shape_dt(j, q) *
                            fe_values_spacetime.JxW(q);

                          // :math:`(\nabla u, \nabla v)`
                          cell_matrix(i + n * dofs_per_spacetime_cell,
                                      j + n * dofs_per_spacetime_cell) +=
                            fe_values_spacetime.shape_space_grad(i, q) *
                            fe_values_spacetime.shape_space_grad(j, q) *
                            fe_values_spacetime.JxW(q);

                        } // dofs j

                      cell_rhs(i + n * dofs_per_spacetime_cell) +=
                        fe_values_spacetime.shape_value(i, q) *
                        right_hand_side.value(x_q) * fe_values_spacetime.JxW(q);

                    } // dofs i

                } // quad

              for (unsigned int q = 0; q < n_quad_space; ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_spacetime_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_spacetime_cell; ++j)
                        {
                          // :math:`(u^+, v^+)`
                          cell_matrix(i + n * dofs_per_spacetime_cell,
                                      j + n * dofs_per_spacetime_cell) +=
                            fe_jump_values_spacetime.shape_value_plus(i, q) *
                            fe_jump_values_spacetime.shape_value_plus(j, q) *
                            fe_jump_values_spacetime.JxW(q);

                          // :math:`-(u^-, v^+)`
                          if (n > 0)
                            {
                              cell_matrix(i + n * dofs_per_spacetime_cell,
                                          j + (n - 1) *
                                                dofs_per_spacetime_cell) -=
                                fe_jump_values_spacetime.shape_value_plus(i,
                                                                          q) *
                                fe_jump_values_spacetime.shape_value_minus(j,
                                                                           q) *
                                fe_jump_values_spacetime.JxW(q);
                            }
                        } // dofs j
                      // :math:`-(u_0,v^+)`
                      if (n == 0)
                        {
                          cell_rhs(i) +=
                            fe_jump_values_spacetime.shape_value_plus(i, q) *
                            initial_values[q] // value of previous solution
                            * fe_jump_values_spacetime.JxW(q);
                        }
                    } // dofs i
                }

            } // cell time
          slab_constraints->distribute_local_to_global(
            cell_matrix,
            cell_rhs,
            local_spacetime_dof_index,
            slab_system_matrix,
            slab_system_rhs);
        }
    } // cell space

  // We need to communicate local contributions to other processors after
  // assembly
  slab_system_matrix.compress(dealii::VectorOperation::add);
  slab_system_rhs.compress(dealii::VectorOperation::add);
}

////////////////////////////////////////////
// @<H3> Step3::solve_system_on_slab
////////////////////////////////////////////
//
void
Step3::solve_system_on_slab()
{
  // The interface needs a solver control that is practically irrelevant
  // as we use a direct solver without convergence criterion
  dealii::SolverControl sc(10000, 1.0e-14, false, false);
  // Amesos_Klu is always installed with Trilinos Amesos.
  // Note that both SuperLU_dist and MUMPS might have a better performance.
  dealii::TrilinosWrappers::SolverDirect::AdditionalData ad(false,
                                                            "Amesos_Klu");
  dealii::TrilinosWrappers::SolverDirect                 solver(sc, ad);
  // In this interface the factorization step is called initialize
  solver.initialize(slab_system_matrix);
  solver.solve(slab_owned_tmp, slab_system_rhs);
  slab_constraints->distribute(slab_owned_tmp);
  // The solver can only write into a vector with locally owned dofs
  // so we need to communicate that result between the processors
  *slab_its.solution = slab_owned_tmp;
}

////////////////////////////////////////////
// @<H3> Step3::process_results_on_slab
////////////////////////////////////////////
// Here, we add the local contribution to
// :math:`(u-u_{kh},u-u_{kh})_{L^2(Q)}=||u-u_{kh}||_{L^2(Q)}^2`.
void
Step3::process_results_on_slab()
{
  idealii::spacetime::QGauss<2> quad(fe.spatial()->degree + 2,
                                     fe.temporal()->degree + 2);
  L2_sqr_error +=
    idealii::slab::VectorTools::calculate_L2L2_squared_error_on_slab<2>(
      *slab_its.dof, *slab_its.solution, exact_solution, quad

    );
}
////////////////////////////////////////////
// @<H3> Step3::output_results_on_slab
////////////////////////////////////////////
//
void
Step3::output_results_on_slab()
{
  auto n_dofs = slab_its.dof->n_dofs_time();

  // To distinguish output between the different support types we append the
  // name to the output filename
  std::string support_type;
  if (fe.type() == DGFE::support_type::Lobatto)
    support_type = "Lobatto";
  else if (fe.type() == DGFE::support_type::Legendre)
    support_type = "Legendre";
  else if (fe.type() == DGFE::support_type::RadauLeft)
    support_type = "RadauLeft";
  else
    support_type = "RadauRight";

  dealii::TrilinosWrappers::MPI::Vector tmp = *slab_its.solution;
  for (unsigned i = 0; i < n_dofs; i++)
    {
      dealii::DataOut<2> data_out;
      data_out.attach_dof_handler(*slab_its.dof->spatial());
      dealii::TrilinosWrappers::MPI::Vector local_solution;
      local_solution.reinit(space_locally_owned_dofs,
                            space_locally_relevant_dofs,
                            mpi_comm);

      idealii::slab::VectorTools::extract_subvector_at_time_dof(tmp,
                                                                local_solution,
                                                                i);
      data_out.add_data_vector(local_solution, "Solution");
      data_out.build_patches();
      std::ostringstream filename;
      filename << "solution_" << support_type << "_cG(" << fe.spatial()->degree
               << ")dG(" << fe.temporal()->degree << ")_t_" << slab * n_dofs + i
               << ".vtu";
      // instead of a vtk we use the parallel write function
      data_out.write_vtu_in_parallel(filename.str().c_str(), mpi_comm);
    }
}

////////////////////////////////////////////
// @<H2> The main function
////////////////////////////////////////////
// We want to be able to do paramter studies without having to recompile the
// whole program. An option would be the ParameterHandler class of deal.II,
// but passing command line arguments makes the study easier to automate.
int
main(int argc, char *argv[])
{
  // With MPI we need to begin with an InitFinalize call.
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  // Trilinos Teuchos offers a nice interface to specify and parse command line
  // arguments.
  // Single options are added using one of the various ``setOption`` methods.
  Teuchos::CommandLineProcessor clp;
  clp.setDocString("This example program demonstrates solving the heat "
                   "equation with Trilinos + MPI");
  bool write_vtu = true;
  clp.setOption("write-vtu",
                "no-vtu",
                &write_vtu,
                "Write results into vtu files?");
  // Finite element orders for cG(s)dG(r) element.
  int s = 1;
  int r = 0;
  clp.setOption("s", &s, "spatial FE degree");
  clp.setOption("r", &r, "temporal FE degree");
  // Number of temporal elements
  int M = 100;
  clp.setOption("M", &M, "Number of temporal elements");
  // Number of spatial refinements resulting in :math:`h = 0.5^(n_{\text{ref})`
  int n_ref_space = 6;
  clp.setOption("n-ref-space", &n_ref_space, "Number of spatial refinements");
  // Which temporal support type do we want to use?
  DGFE::support_type       st          = DGFE::support_type::Lobatto;
  const DGFE::support_type st_values[] = {DGFE::support_type::Lobatto,
                                          DGFE::support_type::Legendre,
                                          DGFE::support_type::RadauLeft,
                                          DGFE::support_type::RadauRight};
  const char *st_names[] = {"Lobatto", "Legendre", "RadauLeft", "RadauRight"};
  clp.setOption("support-type",
                &st,
                4,
                st_values,
                st_names,
                "Location of temporal FE support points");

  clp.throwExceptions(false);

  Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_return =
    clp.parse(argc, argv);
  if (parse_return == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED)
    {
      return 0; // don't fail if the program was called with ``--help``.
    }
  if (parse_return != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      return 1; // Error!
    }

  Step3 problem(s, r, M, n_ref_space, st, write_vtu);
  problem.run();
}
