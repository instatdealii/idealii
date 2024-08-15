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
// Compared to step-2 for coupled problems and step-3 for MPI-parallel linear
// algebra with Trilinos, there are no new includes in this tutorial step.

////////////////////////////////////////////
// @<H3> ideal.II includes
////////////////////////////////////////////
//
#include <ideal.II/base/quadrature_lib.hh>
#include <ideal.II/base/time_iterator.hh>

#include <ideal.II/distributed/fixed_tria.hh>

#include <ideal.II/dofs/slab_dof_tools.hh>
#include <ideal.II/dofs/spacetime_dof_handler.hh>

#include <ideal.II/fe/fe_dg.hh>
#include <ideal.II/fe/spacetime_fe_values.hh>

#include <ideal.II/lac/spacetime_trilinos_vector.hh>

#include <ideal.II/numerics/vector_tools.hh>

////////////////////////////////////////////
// @<H3> deal.II includes
////////////////////////////////////////////
//
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

////////////////////////////////////////////
// @<H3> Trilinos and C++ includes
////////////////////////////////////////////
//
#include <Teuchos_CommandLineProcessor.hpp>

#include <fstream>
#include <vector>

////////////////////////////////////////////
// @<H2> Space-time functions
////////////////////////////////////////////
// The inflow condition is the same as in step-2
class PoisseuilleInflow : public dealii::Function<2>
{
public:
  PoisseuilleInflow(double max_inflow_velocity = 1.5,
                    double channel_height      = 0.41)
    : Function<2>(3)
    , max_inflow_velocity(max_inflow_velocity)
    , H(channel_height)
  {}

  virtual double
  value(const dealii::Point<2> &p, const unsigned int component = 0) const;

  virtual void
  vector_value(const dealii::Point<2> &p, dealii::Vector<double> &value) const;

private:
  double max_inflow_velocity;
  double H;
};

double
PoisseuilleInflow::value(const dealii::Point<2> &p,
                         const unsigned int      component) const
{
  Assert(component < this->n_components,
         dealii::ExcIndexRange(component,
                               0,
                               this->n_components)) if (component == 0)
  {
    double y = p(1);
    if (p(0) == 0 && y <= 0.41)
      {
        double t = get_time();
        return 4 * max_inflow_velocity * y * (H - y) *
               std::sin(M_PI * t * 0.125) / (H * H);
      }
  }
  return 0;
}

void
PoisseuilleInflow::vector_value(const dealii::Point<2> &p,
                                dealii::Vector<double> &values) const
{
  for (unsigned int c = 0; c < this->n_components; c++)
    {
      values(c) = value(p, c);
    }
}


////////////////////////////////////////////
// @<H2> The Step4 class
////////////////////////////////////////////
// This class describes the solution of the nonlinear Navier-Stokes equations
// with space-time slab tensor-product elements and a Newton linearization.
// Compared to earlier steps we have a few new functions
// We split the assembly into matrix and right hand side, as we have to call the
// latter more frequently in the Newton algorithm.
// Additionally, we now have functions to calculate the functionals,
// i.e. pressure as well as drag and lift.
class Step4
{
public:
  Step4(unsigned int temporal_degree, bool write_vtu);
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
  assemble_residual_on_slab();
  void
  assemble_system_on_slab();
  void
  solve_Newton_problem_on_slab();
  void
  calculate_functional_values_on_slab();
  double
  calculate_pressure_at_point(const dealii::Point<2>                       x,
                              const dealii::TrilinosWrappers::MPI::Vector &u);
  void
  calculate_drag_lift_tensor(dealii::TrilinosWrappers::MPI::Vector &u,
                             dealii::Tensor<1, 2> &drag_lift_value);

  void
  output_results_on_slab();

  MPI_Comm mpi_comm; // MPI communicator
  bool     write_vtu;
  // We want to write out the temporal trend of the functional values
  // into a file and for that log we need to construct a name as well.
  std::ofstream      functional_log;
  std::ostringstream logname;
  // We need the kinematic viscosity at multiple spots so we add it as a member
  // variable to ensure consistency.
  double nu_f;

  dealii::ConditionalOStream pout;

  idealii::spacetime::parallel::distributed::fixed::Triangulation<2>
                                     triangulation;
  idealii::spacetime::DoFHandler<2>  dof_handler;
  idealii::spacetime::TrilinosVector solution;

  idealii::spacetime::DG_FiniteElement<2> fe;

  dealii::SparsityPattern                            slab_sparsity_pattern;
  std::shared_ptr<dealii::AffineConstraints<double>> slab_zero_constraints;
  dealii::TrilinosWrappers::SparseMatrix             slab_system_matrix;
  dealii::TrilinosWrappers::MPI::Vector              slab_system_rhs;
  dealii::TrilinosWrappers::MPI::Vector              slab_initial_value;
  unsigned int                                       slab;

  dealii::TrilinosWrappers::MPI::Vector slab_newton_update;
  dealii::TrilinosWrappers::MPI::Vector slab_owned_tmp;
  dealii::TrilinosWrappers::MPI::Vector slab_relevant_tmp;

  dealii::IndexSet slab_locally_owned_dofs;
  dealii::IndexSet slab_locally_relevant_dofs;

  dealii::IndexSet space_locally_owned_dofs;
  dealii::IndexSet space_locally_relevant_dofs;

  struct
  {
    idealii::slab::parallel::distributed::TriaIterator<2> tria;
    idealii::slab::DoFHandlerIterator<2>                  dof;
    idealii::slab::TrilinosVectorIterator                 solution;
  } slab_its;
};

////////////////////////////////////////////
// @<H3> Step4::Step4
////////////////////////////////////////////
//
Step4::Step4(unsigned int temporal_degree, bool write_vtu)
  : mpi_comm(MPI_COMM_WORLD)
  , write_vtu(write_vtu)
  , nu_f(1.0e-3)
  , pout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  , triangulation()
  , dof_handler(&triangulation)
  , fe(std::make_shared<dealii::FESystem<2>>(dealii::FE_Q<2>(2),
                                             2,
                                             dealii::FE_Q<2>(1),
                                             1),
       temporal_degree,
       idealii::spacetime::DG_FiniteElement<2>::support_type::Legendre)
  , slab(0)
{
  // This is the easiest place to add the temporal degree
  // to the filename of the functional log, so we do that.
  logname << "functional_log_dG" << temporal_degree;
}

////////////////////////////////////////////
// @<H3> Step4::run
////////////////////////////////////////////
//
void
Step4::run() // same as before
{
  make_grid();
  time_marching();
}

////////////////////////////////////////////
// @<H3> Step4::make_grid
////////////////////////////////////////////
//
void
Step4::make_grid()
{
  auto space_tria = // construct an MPI parallel triangulation
    std::make_shared<dealii::parallel::distributed::Triangulation<2>>(mpi_comm);
  dealii::GridIn<2> grid_in;
  grid_in.attach_triangulation(*space_tria);
  std::ifstream input_file("nsbench4.inp");
  grid_in.read_ucd(input_file);
  dealii::Point<2>                          p(0.2, 0.2);
  static const dealii::SphericalManifold<2> boundary(p);
  dealii::GridTools::copy_boundary_to_manifold_id(*space_tria);
  space_tria->set_manifold(80, boundary);
  const unsigned int M = 256;
  triangulation.generate(space_tria, M, 0, 8.);
  const unsigned int n_ref_space = 2;
  triangulation.refine_global(n_ref_space, 0);
  dof_handler.generate();
  // To have the full information in the log we also add the number of
  // temporal DoFs and the number of spatial refinements to the
  // filename.
  logname << "_M" << M << "_lvl" << n_ref_space << ".csv";
}

////////////////////////////////////////////
// @<H3> Step4::time_marching
////////////////////////////////////////////
//
void
Step4::time_marching()
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

  // We don't want to write each time DoF information multiple times,
  // so we only open the file on rank 0.
  // We also write out a header for the CSV file so it is clear
  // which column will be which variable.
  if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::cout << "Opening functional log file: " << logname.str()
                << std::endl;
      functional_log.open(logname.str());
      functional_log << "t, pfront, pback, pdiff, drag, lift" << std::endl;
    }

  pout << "*******Starting time-stepping*********" << std::endl;
  for (; !tic.at_end(); tic.increment())
    {
      pout << "Starting time-step (" << slab_its.tria->startpoint() << ","
           << slab_its.tria->endpoint() << ")" << std::endl;

      setup_system_on_slab();
      // As the system matrix for a nonlinear problem depends on the
      // approximate solution, the assembly is done within the Newton
      // iterations.
      solve_Newton_problem_on_slab();

      calculate_functional_values_on_slab();

      if (write_vtu)
        output_results_on_slab();

      // As in step-3 we need to extract the solution at the final time point as
      // the final DoF is not in the same location for the chosen
      // Legendre support points.
      idealii::slab::VectorTools::extract_subvector_at_time_point(
        *slab_its.dof,
        *slab_its.solution,
        slab_initial_value,
        slab_its.tria->endpoint());
      slab++;
    }
  if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      functional_log.close();
    }
}

////////////////////////////////////////////
// @<H3> Step4::setup_system_on_slab
////////////////////////////////////////////
//
void
Step4::setup_system_on_slab()
{
  slab_its.dof->distribute_dofs(fe);

  // If we renumber we get a problem with parallel data output,
  // so we omit that here!

  pout << "Number of degrees of freedom: \n\t" << slab_its.dof->n_dofs_space()
       << " (space) * " << slab_its.dof->n_dofs_time()
       << " (time) = " << slab_its.dof->n_dofs_spacetime() << std::endl;

  // As in step-3 we need to know which DoFs are owned by the current MPI rank
  // and which are ghost points beloging to a different rank but are also part
  // of an owned spatial element.
  space_locally_owned_dofs = slab_its.dof->spatial()->locally_owned_dofs();
  dealii::DoFTools::extract_locally_relevant_dofs(*slab_its.dof->spatial(),
                                                  space_locally_relevant_dofs);

  slab_locally_owned_dofs = slab_its.dof->locally_owned_dofs();
  slab_locally_relevant_dofs =
    idealii::slab::DoFTools::extract_locally_relevant_dofs(*slab_its.dof);

  slab_owned_tmp.reinit(slab_locally_owned_dofs, mpi_comm);
  slab_relevant_tmp.reinit(slab_locally_owned_dofs,
                           slab_locally_relevant_dofs,
                           mpi_comm);
  slab_newton_update.reinit(slab_locally_owned_dofs, mpi_comm);
  if (slab == 0)
    {
      slab_initial_value.reinit(space_locally_owned_dofs,
                                space_locally_relevant_dofs,
                                mpi_comm);
      slab_initial_value = 0;
    }

  // We need two sets of constraints for the defect correction Newton method.
  // The Newton updates will be added to the current solution in each step.
  // So in order to leave the boundary conditions of the solution unchanged,
  // the update will need to have zero Dirichlet boundary conditions.
  // Consequently, the initial guess will need to have correct boundary
  // conditions.
  slab_zero_constraints = std::make_shared<dealii::AffineConstraints<double>>();
  auto slab_initial_constraints =
    std::make_shared<dealii::AffineConstraints<double>>();
  auto              zero   = dealii::Functions::ZeroFunction<2>(3);
  auto              inflow = PoisseuilleInflow();
  std::vector<bool> component_mask(3, true);
  component_mask[2] = false;

  idealii::slab::VectorTools::interpolate_boundary_values(
    *slab_its.dof, 0, inflow, slab_initial_constraints, component_mask);

  idealii::slab::VectorTools::interpolate_boundary_values(
    *slab_its.dof, 2, zero, slab_initial_constraints, component_mask);

  idealii::slab::VectorTools::interpolate_boundary_values(
    *slab_its.dof, 80, zero, slab_initial_constraints, component_mask);

  slab_initial_constraints->close();

  idealii::slab::VectorTools::interpolate_boundary_values(
    *slab_its.dof, 0, zero, slab_zero_constraints, component_mask);

  idealii::slab::VectorTools::interpolate_boundary_values(
    *slab_its.dof, 2, zero, slab_zero_constraints, component_mask);

  idealii::slab::VectorTools::interpolate_boundary_values(
    *slab_its.dof, 80, zero, slab_zero_constraints, component_mask);

  slab_zero_constraints->close();

  // Since the Newton method is iterative we have to start with some initial
  // guess and the convergence speed depends on the closeness of the
  // current approximation to the correct solution.
  // Without prior knowledge this initial guess is often zero.
  // Here, however we know the solution at the last temporal DoF of the
  // previous slab and for a fine enough temporal mesh that should at least
  // be closer to the correct solution than zero and so we choose that
  // and only correct the guess at the Dirichlet boundaries.
  dealii::IndexSet::ElementIterator lri = space_locally_owned_dofs.begin();
  dealii::IndexSet::ElementIterator lre = space_locally_owned_dofs.end();
  for (; lri != lre; lri++)
    {
      for (unsigned int ii = 0; ii < slab_its.dof->n_dofs_time(); ii++)
        {
          slab_owned_tmp[*lri + slab_its.dof->n_dofs_space() * ii] =
            slab_initial_value[*lri];
        }
    }
  slab_initial_constraints->distribute(slab_owned_tmp);

  // The spatial pressure-pressure coupling block is empty
  // and the sparsity pattern can be empty in these blocks as in step-2.
  dealii::Table<2, dealii::DoFTools::Coupling> coupling_space(2 + 1, 2 + 1);
  for (unsigned int i = 0; i < 2; i++)
    {
      coupling_space[i][2] = dealii::DoFTools::always;
      coupling_space[2][i] = dealii::DoFTools::always;
      for (unsigned int j = 0; j < 2; j++)
        {
          coupling_space[i][j] = dealii::DoFTools::always;
        }
    }

  dealii::DynamicSparsityPattern dsp(slab_its.dof->n_dofs_spacetime());
  idealii::slab::DoFTools::make_upwind_sparsity_pattern(*slab_its.dof,
                                                        coupling_space,
                                                        dsp,
                                                        slab_zero_constraints);

  dealii::SparsityTools::distribute_sparsity_pattern(
    dsp, slab_locally_owned_dofs, mpi_comm, slab_locally_relevant_dofs);


  slab_system_matrix.reinit(slab_locally_owned_dofs,
                            slab_locally_owned_dofs,
                            dsp);

  slab_its.solution->reinit(slab_locally_owned_dofs,
                            slab_locally_relevant_dofs,
                            mpi_comm);
  slab_system_rhs.reinit(slab_locally_owned_dofs, mpi_comm);
  *slab_its.solution = slab_owned_tmp;
}

////////////////////////////////////////////
// @<H3> Step4::assemble_system_on_slab
////////////////////////////////////////////
// In contrast to previous steps we only assemble the matrix here.
// This is because we have to assemble the right hand side, i.e. Newton residual
// more often.
void
Step4::assemble_system_on_slab()
{
  slab_system_matrix = 0;
  // We use a higher spatial quadrature order, since the nonlinear convection
  // term is of higher order.
  idealii::spacetime::QGauss<2> quad(fe.spatial()->degree + 3,
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

  const unsigned int dofs_per_spacetime_cell = fe.dofs_per_cell;

  auto N = slab_its.tria->temporal()->n_global_active_cells();
  dealii::FullMatrix<double> cell_matrix(N * dofs_per_spacetime_cell,
                                         N * dofs_per_spacetime_cell);

  std::vector<dealii::types::global_dof_index> local_spacetime_dof_index(
    N * dofs_per_spacetime_cell);

  unsigned int n;
  unsigned int n_quad_spacetime = fe_values_spacetime.n_quadrature_points;
  unsigned int n_quad_space     = quad.spatial()->size();

  dealii::FEValuesExtractors::Vector velocity(0);
  dealii::FEValuesExtractors::Scalar pressure(2);

  // The convection term includes the previous Newton iterate and its
  // derivative, so we have to allocate storage to evaluate the iterate at each
  // quadrature point.

  std::vector<dealii::Vector<double>> old_solution_values(
    n_quad_spacetime, dealii::Vector<double>(3));

  std::vector<std::vector<dealii::Tensor<1, 2>>> old_solution_grads(
    n_quad_spacetime, std::vector<dealii::Tensor<1, 2>>(3));

  for (const auto &cell_space :
       slab_its.dof->spatial()->active_cell_iterators())
    {
      if (cell_space->is_locally_owned()) // only rank local contributions
        {
          fe_values_spacetime.reinit_space(cell_space);
          fe_jump_values_spacetime.reinit_space(cell_space);

          cell_matrix = 0;

          for (const auto &cell_time :
               slab_its.dof->temporal()->active_cell_iterators())
            {
              n = cell_time->index();
              fe_values_spacetime.reinit_time(cell_time);
              fe_jump_values_spacetime.reinit_time(cell_time);
              fe_values_spacetime.get_local_dof_indices(
                local_spacetime_dof_index);

              // evaluate the previous Newton iterate on the current space-time
              // element.
              fe_values_spacetime.get_function_values(*slab_its.solution,
                                                      old_solution_values);

              fe_values_spacetime.get_function_space_gradients(
                *slab_its.solution, old_solution_grads);

              for (unsigned int q = 0; q < n_quad_spacetime; ++q)
                {
                  // We save the previous Newton iterate at the current
                  // quadrature point into more useful data structures for the
                  // assembly.
                  dealii::Tensor<1, 2> v;
                  dealii::Tensor<2, 2> grad_v;
                  for (int c = 0; c < 2; c++)
                    {
                      v[c] = old_solution_values[q](c);
                      for (int d = 0; d < 2; d++)
                        {
                          grad_v[c][d] = old_solution_grads[q][c][d];
                        }
                    }
                  for (unsigned int i = 0; i < dofs_per_spacetime_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_spacetime_cell; ++j)
                        {
                          // (convection term)
                          cell_matrix(i + n * dofs_per_spacetime_cell,
                                      j + n * dofs_per_spacetime_cell) +=
                            fe_values_spacetime.vector_value(velocity, i, q) *
                            (fe_values_spacetime.vector_space_grad(velocity,
                                                                   j,
                                                                   q) *
                               v +
                             grad_v * fe_values_spacetime.vector_value(velocity,
                                                                       j,
                                                                       q)) *
                            fe_values_spacetime.JxW(q);

                          // :math:`(\partial_t u,v)`
                          cell_matrix(i + n * dofs_per_spacetime_cell,
                                      j + n * dofs_per_spacetime_cell) +=
                            fe_values_spacetime.vector_value(velocity, i, q) *
                            fe_values_spacetime.vector_dt(velocity, j, q) *
                            fe_values_spacetime.JxW(q);

                          // :math:`(\nabla u, \nabla v)`
                          cell_matrix(i + n * dofs_per_spacetime_cell,
                                      j + n * dofs_per_spacetime_cell) +=
                            dealii::scalar_product(
                              fe_values_spacetime.vector_space_grad(velocity,
                                                                    i,
                                                                    q),
                              fe_values_spacetime.vector_space_grad(velocity,
                                                                    j,
                                                                    q)) *
                            fe_values_spacetime.JxW(q) * nu_f;

                          // :math:`-(p,\nabla\cdot v)`
                          cell_matrix(i + n * dofs_per_spacetime_cell,
                                      j + n * dofs_per_spacetime_cell) -=
                            fe_values_spacetime.vector_divergence(velocity,
                                                                  i,
                                                                  q) *
                            fe_values_spacetime.scalar_value(pressure, j, q) *
                            fe_values_spacetime.JxW(q);

                          // :math:`(\nabla\cdot u,q)`
                          cell_matrix(i + n * dofs_per_spacetime_cell,
                                      j + n * dofs_per_spacetime_cell) +=
                            fe_values_spacetime.scalar_value(pressure, i, q) *
                            fe_values_spacetime.vector_divergence(velocity,
                                                                  j,
                                                                  q) *
                            fe_values_spacetime.JxW(q);

                        } // dofs j

                    } // dofs i

                } // quad

              for (unsigned int q = 0; q < n_quad_space; ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_spacetime_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_spacetime_cell; ++j)
                        {
                          // (u^+, v^+)
                          cell_matrix(i + n * dofs_per_spacetime_cell,
                                      j + n * dofs_per_spacetime_cell) +=
                            fe_jump_values_spacetime.vector_value_plus(velocity,
                                                                       i,
                                                                       q) *
                            fe_jump_values_spacetime.vector_value_plus(velocity,
                                                                       j,
                                                                       q) *
                            fe_jump_values_spacetime.JxW(q);

                          // -(u^-, v^+)
                          if (n > 0)
                            {
                              cell_matrix(i + n * dofs_per_spacetime_cell,
                                          j + (n - 1) *
                                                dofs_per_spacetime_cell) -=
                                fe_jump_values_spacetime.vector_value_plus(
                                  velocity, i, q) *
                                fe_jump_values_spacetime.vector_value_minus(
                                  velocity, j, q) *
                                fe_jump_values_spacetime.JxW(q);
                            }
                        } // dofs j

                    } // dofs i
                }

            } // cell time
          slab_zero_constraints->distribute_local_to_global(
            cell_matrix, local_spacetime_dof_index, slab_system_matrix);
        }
    } // cell space

  slab_system_matrix.compress(dealii::VectorOperation::add); // communication
}

////////////////////////////////////////////
// @<H3> Step4::assemble_residual_on_slab
////////////////////////////////////////////
//
// For the Newton iteration we also need to assemble the residual vector
// which also depends on the current slab solution/iterate.

void
Step4::assemble_residual_on_slab()
{
  slab_system_rhs = 0;
  idealii::spacetime::QGauss<2> quad(fe.spatial()->degree + 3,
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

  const unsigned int dofs_per_spacetime_cell = fe.dofs_per_cell;

  auto                   N = slab_its.tria->temporal()->n_global_active_cells();
  dealii::Vector<double> cell_rhs(N * dofs_per_spacetime_cell);

  std::vector<dealii::types::global_dof_index> local_spacetime_dof_index(
    N * dofs_per_spacetime_cell);

  unsigned int n;
  unsigned int n_quad_spacetime = fe_values_spacetime.n_quadrature_points;
  unsigned int n_quad_space     = quad.spatial()->size();

  dealii::FEValuesExtractors::Vector velocity(0);
  dealii::FEValuesExtractors::Scalar pressure(2);

  // Compared to the assembly of the system matrix we have to also evaluate the
  // temporal derivative and the limits at the temporal element edges of the
  // previous Newton iterate.
  std::vector<dealii::Vector<double>> old_solution_values(
    n_quad_spacetime, dealii::Vector<double>(3));

  std::vector<dealii::Vector<double>> old_solution_dt(
    n_quad_spacetime, dealii::Vector<double>(3));

  std::vector<std::vector<dealii::Tensor<1, 2>>> old_solution_grads(
    n_quad_spacetime, std::vector<dealii::Tensor<1, 2>>(3));

  std::vector<dealii::Vector<double>> old_solution_plus(
    n_quad_space, dealii::Vector<double>(3));

  std::vector<dealii::Vector<double>> old_solution_minus(
    n_quad_space, dealii::Vector<double>(3));

  for (const auto &cell_space :
       slab_its.dof->spatial()->active_cell_iterators())
    {
      if (cell_space->is_locally_owned()) // only rank local contributions
        {
          fe_values_spacetime.reinit_space(cell_space);
          fe_jump_values_spacetime.reinit_space(cell_space);

          cell_rhs = 0;
          for (const auto &cell_time :
               slab_its.dof->temporal()->active_cell_iterators())
            {
              n = cell_time->index();
              fe_values_spacetime.reinit_time(cell_time);
              fe_jump_values_spacetime.reinit_time(cell_time);
              fe_values_spacetime.get_local_dof_indices(
                local_spacetime_dof_index);

              fe_values_spacetime.get_function_values(*slab_its.solution,
                                                      old_solution_values);

              fe_values_spacetime.get_function_dt(*slab_its.solution,
                                                  old_solution_dt);

              fe_values_spacetime.get_function_space_gradients(
                *slab_its.solution, old_solution_grads);

              for (unsigned int q = 0; q < n_quad_spacetime; ++q)
                {
                  dealii::Tensor<1, 2> v;
                  dealii::Tensor<1, 2> dt_v;
                  dealii::Tensor<2, 2> grad_v;

                  const double p = old_solution_values[q](2);
                  for (int c = 0; c < 2; c++)
                    {
                      v[c]    = old_solution_values[q](c);
                      dt_v[c] = old_solution_dt[q](c);
                      for (int d = 0; d < 2; d++)
                        {
                          grad_v[c][d] = old_solution_grads[q][c][d];
                        }
                    }
                  const double div_v = dealii::trace(grad_v);

                  for (unsigned int i = 0; i < dofs_per_spacetime_cell; ++i)
                    {
                      // (dt u, v)
                      cell_rhs(i + n * dofs_per_spacetime_cell) -=
                        fe_values_spacetime.vector_value(velocity, i, q) *
                        dt_v * fe_values_spacetime.JxW(q);

                      // convection
                      cell_rhs(i + n * dofs_per_spacetime_cell) -=
                        fe_values_spacetime.vector_value(velocity, i, q) *
                        grad_v * v * fe_values_spacetime.JxW(q);

                      // (grad u, grad v)
                      cell_rhs(i + n * dofs_per_spacetime_cell) -=
                        nu_f *
                        dealii::scalar_product(
                          fe_values_spacetime.vector_space_grad(velocity, i, q),
                          grad_v) *
                        fe_values_spacetime.JxW(q);

                      // (pressure gradient)
                      cell_rhs(i + n * dofs_per_spacetime_cell) +=
                        fe_values_spacetime.vector_divergence(velocity, i, q) *
                        p * fe_values_spacetime.JxW(q);

                      // (div free constraint)
                      cell_rhs(i + n * dofs_per_spacetime_cell) -=
                        fe_values_spacetime.scalar_value(pressure, i, q) *
                        div_v * fe_values_spacetime.JxW(q);

                    } // dofs i

                } // quad

              if (n == 0)
                {
                  fe_values_spacetime.spatial()->get_function_values(
                    slab_initial_value, old_solution_minus);
                }

              fe_jump_values_spacetime.get_function_values_plus(
                *slab_its.solution, old_solution_plus);

              dealii::Tensor<1, 2> v_plus;
              dealii::Tensor<1, 2> v_minus;

              for (unsigned int q = 0; q < n_quad_space; ++q)
                {
                  for (unsigned int c = 0; c < 2; ++c)
                    {
                      v_plus[c]  = old_solution_plus[q](c);
                      v_minus[c] = old_solution_minus[q](c);
                    }

                  for (unsigned int i = 0; i < dofs_per_spacetime_cell; ++i)
                    {
                      // :math:`(u^+, v^+)`
                      cell_rhs(i + n * dofs_per_spacetime_cell) -=
                        fe_jump_values_spacetime.vector_value_plus(velocity,
                                                                   i,
                                                                   q) *
                        v_plus * fe_jump_values_spacetime.JxW(q);

                      // :math:`-(u^-, v^+)`
                      cell_rhs(i + n * dofs_per_spacetime_cell) +=
                        fe_jump_values_spacetime.vector_value_plus(velocity,
                                                                   i,
                                                                   q) *
                        v_minus * fe_jump_values_spacetime.JxW(q);

                    } // dofs i

                } // quad_space
              if (n < N - 1)
                {
                  fe_jump_values_spacetime.get_function_values_minus(
                    *slab_its.solution, old_solution_minus);
                }
            } // cell time
          slab_zero_constraints->distribute_local_to_global(
            cell_rhs, local_spacetime_dof_index, slab_system_rhs);
        }
    } // cell space

  // We need to communicate local contributions to other processors after
  // assembly
  slab_system_rhs.compress(dealii::VectorOperation::add);
}

///////////////////////////////////////////////
// @<H3> Step4::solve_Newton_problem_on_slab
///////////////////////////////////////////////
//
void
Step4::solve_Newton_problem_on_slab()
{
  pout << "Starting Newton solve" << std::endl;

  // We use a direct solver as the inner solver for each Newton iterate.
  // Again, if you installed Trilinos with other third party solvers you can
  // change them, e.g. to MUMPS or SuperLU_dist.
  dealii::SolverControl sc(10000, 1.0e-14, false, false);
  dealii::TrilinosWrappers::SolverDirect::AdditionalData ad(false,
                                                            "Amesos_Klu");

  auto solver =
    std::make_shared<dealii::TrilinosWrappers::SolverDirect>(sc, ad);

  // Newton parameters
  double       newton_lower_bound       = 1.0e-10;
  unsigned int max_newton_steps         = 10;
  unsigned int max_line_search_steps    = 10;
  double       newton_rebuild_parameter = 0.1;
  double       newton_damping           = 0.6;

  assemble_residual_on_slab();
  double newton_residual = slab_system_rhs.linfty_norm();
  double old_newton_residual;
  double new_newton_residual;

  unsigned int newton_step = 1;
  unsigned int line_search_step;

  pout << "0\t" << newton_residual << std::endl;

  // We iterate either until the newton residual is small enough,
  // or until we have reached the maxmimum number of Newton steps.
  while (newton_residual > newton_lower_bound &&
         newton_step <= max_newton_steps)
    {
      old_newton_residual = newton_residual;
      assemble_residual_on_slab();
      newton_residual = slab_system_rhs.linfty_norm();

      if (newton_residual < newton_lower_bound)
        {
          pout << "res\t" << newton_residual << std::endl;
          break;
        }

      // If we do not have enough reduction from the previous step,
      // we assemble a new system matrix with the most current iterate.
      if (newton_residual / old_newton_residual > newton_rebuild_parameter)
        {
          solver = nullptr;
          solver =
            std::make_shared<dealii::TrilinosWrappers::SolverDirect>(sc, ad);
          assemble_system_on_slab();

          // Since we are using a direct solver we only have to do the costly
          // factorization once after assembling a new matrix.
          solver->initialize(slab_system_matrix);
        }

      // Having an existing factorization we only have to do the actual solve,
      // i.e. two triangular matrix solves.
      // Note that we solve for an update, i.e. :math:`\delta U` instead of
      // :math:`U` directly.
      solver->solve(slab_newton_update, slab_system_rhs);
      // We have to make sure the boundary values are correct.
      slab_zero_constraints->distribute(slab_newton_update);
      // Finally we update the ghost values in the temporary vector.
      slab_owned_tmp = *slab_its.solution;

      // Now, we update our solution with a possibly damped upgrade,
      // i.e. :math:`U^\text{new}=U^\text{old}+\alpha\delta U`.
      // We start by trying a full step (:math:`\alpha=1`) and check if that
      // leads to a reduction in the residual. If that is not the case,
      // we dampen the update by multiplying the step length :math:`\alpha` with
      // ``newton_damping``. We continue dampening until we get a reduction or
      // until we have done ``max_line_search_steps`` dampening steps.
      for (line_search_step = 0; line_search_step < max_line_search_steps;
           line_search_step++)
        {
          slab_owned_tmp += slab_newton_update;
          *slab_its.solution = slab_owned_tmp;
          assemble_residual_on_slab();

          new_newton_residual = slab_system_rhs.linfty_norm();
          if (new_newton_residual < newton_residual)
            break;
          else
            slab_owned_tmp -= slab_newton_update;

          slab_newton_update *= newton_damping;
        }

      // In the following we output information on the current Newton step
      // to console, including the current residual, reduction and whether or
      // not the matrix had to be rebuilt
      pout << std::setprecision(5) << newton_step << "\t" << std::scientific
           << newton_residual << "\t" << std::scientific
           << newton_residual / old_newton_residual << "\t";

      if (newton_residual / old_newton_residual > newton_rebuild_parameter)
        pout << "r\t";
      else
        pout << " \t";

      pout << line_search_step << "\t" << std::scientific << std::endl;

      newton_step++; // Update working index
    }
}

/////////////////////////////////////////////////////
// @<H3> Step4::calculate_functional_values_on_slab
/////////////////////////////////////////////////////
//
void
Step4::calculate_functional_values_on_slab()
{
  // We want to plot the curves of the functional values,
  // so we need to know the time points.
  // The following constructs a dummy quadrature rule that does
  // not have valid weights and is only used to get the support points of the
  // temporal finite element.
  dealii::Quadrature<1> quad_time(
    slab_its.dof->temporal()->get_fe(0).get_unit_support_points());
  // Similarly, our temporal FEValues object is only used to query the support
  // points.
  dealii::FEValues<1> fev(slab_its.dof->temporal()->get_fe(0),
                          quad_time,
                          dealii::update_quadrature_points);

  std::vector<dealii::types::global_dof_index> local_indices(
    slab_its.dof->dofs_per_cell_time());

  auto n_dofs = slab_its.dof->n_dofs_time();

  dealii::TrilinosWrappers::MPI::Vector tmp = *slab_its.solution;

  // allocate storage only once
  double       time     = 0.;
  unsigned int time_dof = 0;
  double       pfront   = 0.;
  double       pback    = 0.;
  double       pdiff    = 0.;

  dealii::Tensor<1, 2> drag_lift_tensor;

  // Points for pressure evaluation
  dealii::Point<2> front(0.15, 0.2);
  dealii::Point<2> back(0.25, 0.2);

  // The vector to extract the solution at a given temporal DoF to
  dealii::TrilinosWrappers::MPI::Vector local_solution;
  local_solution.reinit(space_locally_owned_dofs,
                        space_locally_relevant_dofs,
                        mpi_comm);

  // In contrast to the assembly functions we start with the temporal
  // element loop here as we want to write out the functional values
  // in sequence.
  for (const auto &cell_time :
       slab_its.dof->temporal()->active_cell_iterators())
    {
      fev.reinit(cell_time);
      cell_time->get_dof_indices(local_indices);

      for (unsigned int q = 0; q < quad_time.size(); ++q)
        {
          time     = fev.quadrature_point(q)[0];
          time_dof = local_indices[q];

          idealii::slab::VectorTools::extract_subvector_at_time_dof(
            *slab_its.solution, local_solution, time_dof);

          // The actual functional value calculations are put into seperate
          // functions to make the whole problem easier to extend.
          pfront = calculate_pressure_at_point(front, local_solution);
          pback  = calculate_pressure_at_point(back, local_solution);
          pdiff  = pfront - pback;

          calculate_drag_lift_tensor(local_solution, drag_lift_tensor);

          // We want to write a comma seperated values (CSV) file,
          // so we write out the values for this DoF in a single line.
          functional_log << time << ", " << pfront << ", " << pback << ", "
                         << pdiff << ", " << drag_lift_tensor[0] << ", "
                         << drag_lift_tensor[1] << std::endl;
        }
    }
}

/////////////////////////////////////////////////////
// @<H3> Step4::calculate_pressure_at_point
/////////////////////////////////////////////////////
//
double
Step4::calculate_pressure_at_point(
  const dealii::Point<2>                       x,
  const dealii::TrilinosWrappers::MPI::Vector &u)
{
  // Storage for the local solution at the given point.
  dealii::Vector<double> x_h(3);

  // The evaluation point will only live on a single rank, so we catch the
  // ``ExcPointNotAvailableHere`` exception on all others
  try
    {
      dealii::VectorTools::point_value(*slab_its.dof->spatial(), u, x, x_h);
    }
  catch (typename dealii::VectorTools::ExcPointNotAvailableHere e)
    {}

  // We need to figure out which rank has the nonzero value
  auto minmax = dealii::Utilities::MPI::min_max_avg(x_h[2], mpi_comm);
  // Check if the actual value is negative, i.e. it's absolute value is
  // larger than the maximum (which then is 0)
  if (std::abs(minmax.min) > minmax.max)
    {
      return minmax.min;
    }
  else
    {
      return minmax.max;
    }
}

/////////////////////////////////////////////////////
// @<H3> Step4::calculate_drag_lift_tensor
/////////////////////////////////////////////////////
//
void
Step4::calculate_drag_lift_tensor(dealii::TrilinosWrappers::MPI::Vector &u,
                                  dealii::Tensor<1, 2> &drag_lift_value)
{
  // We want to calculate drag and lift on the obstacle boundary,
  // so we need a face quadrature rule and matching ``FEFaceValues``
  const dealii::QGauss<1> face_quad(6);

  dealii::FEFaceValues<2> fe_face_values(*fe.spatial(),
                                         face_quad,
                                         dealii::update_values |
                                           dealii::update_gradients |
                                           dealii::update_normal_vectors |
                                           dealii::update_JxW_values |
                                           dealii::update_quadrature_points);

  const unsigned int dofs_per_cell   = slab_its.dof->dofs_per_cell_space();
  const unsigned int n_face_q_points = face_quad.size();

  std::vector<unsigned int> local_dof_indices(dofs_per_cell);

  std::vector<dealii::Vector<double>> face_solution_values(
    n_face_q_points, dealii::Vector<double>(3));
  std::vector<std::vector<dealii::Tensor<1, 2>>> face_solution_grads(
    n_face_q_points, std::vector<dealii::Tensor<1, 2>>(3));

  // Allocate storage for :math:`p*I`, :math:`\nabla v` and the stress tensor
  // :math:`-p*I+\nu*\nabla v` needed in the calculation
  dealii::Tensor<2, 2> pI;
  pI.clear();
  dealii::Tensor<2, 2> grad_v;
  dealii::Tensor<2, 2> stress

    drag_lift_value = 0.;

  // We loop over spatial elements only as we call this function from within a
  // temporal loop. We only consider elements that are owned by the current rank
  // and also at a boundary.
  for (const auto &cell : slab_its.dof->spatial()->active_cell_iterators())
    {
      if (cell->is_locally_owned() && cell->at_boundary())
        {
          // On each spatial element we loop over the faces and only consider
          // those that are at the obstacle boundary (id 80)
          for (unsigned int face = 0;
               face < dealii::GeometryInfo<2>::faces_per_cell;
               ++face)
            if (cell->face(face)->at_boundary() &&
                cell->face(face)->boundary_id() == 80)
              {
                fe_face_values.reinit(cell, face);
                fe_face_values.get_function_values(slab_initial_value,
                                                   face_solution_values);
                fe_face_values.get_function_gradients(slab_initial_value,
                                                      face_solution_grads);

                // Loop over quadrature points on the current face
                for (unsigned int q = 0; q < n_face_q_points; ++q)
                  {
                    // Update the values in :math:`p*I` and :math:`\nabla v`
                    for (unsigned int l = 0; l < 2; ++l)
                      {
                        pI[l][l] = face_solution_values[q][2];
                        for (unsigned int m = 0; m < 2; ++m)
                          {
                            grad_v[l][m] = face_solution_grads[q][l][m];
                          }
                      }

                    stress = -pI + nu_f * grad_v;

                    // Add the local contributions to the drag lift tensor
                    drag_lift_value -= stress *
                                       fe_face_values.normal_vector(q) *
                                       fe_face_values.JxW(q);
                  }
              }
        }
    }

  // For now we only have rank local contributions to the drag lift tensor,
  // so we need to sum these up.
  double tmp = dealii::Utilities::MPI::sum(drag_lift_value[0], mpi_comm);
  drag_lift_value[0] = tmp;
  tmp = dealii::Utilities::MPI::sum(drag_lift_value[1], mpi_comm);
  drag_lift_value[1] = tmp;
  // Finally, we have to scale the calculated value.
  drag_lift_value *= 20.;
}
////////////////////////////////////////////
// @<H3> Step4::output_results_on_slab
////////////////////////////////////////////
//
void
Step4::output_results_on_slab() // Nothing new compared to step-2
{
  auto                     n_dofs = slab_its.dof->n_dofs_time();
  std::vector<std::string> field_names;
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    dci;
  for (unsigned int i = 0; i < 2; i++)
    {
      field_names.push_back("velocity");
      dci.push_back(
        dealii::DataComponentInterpretation::component_is_part_of_vector);
    }
  field_names.push_back("pressure");
  dci.push_back(dealii::DataComponentInterpretation::component_is_scalar);

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

      data_out.add_data_vector(local_solution,
                               field_names,
                               dealii::DataOut<2>::type_dof_data,
                               dci);
      data_out.build_patches(2);
      std::ostringstream filename;
      filename << "newton_navierstokes_solution_dG(" << fe.temporal()->degree
               << ")_t_" << slab * n_dofs + i << ".vtu";
      // instead of a vtk we use the parallel write function
      data_out.write_vtu_in_parallel(filename.str().c_str(), mpi_comm);
    }
}

////////////////////////////////////////////
// @<H2> The main function
////////////////////////////////////////////
//
int
main(int argc, char *argv[])
{
  // With MPI we need to begin with an InitFinalize call.
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  // As in step-3 we want to pass command line arguments for a
  // parameter study, but we only want to be able to suppress VTU output
  // and vary the temporal finite element order.
  Teuchos::CommandLineProcessor clp;
  clp.setDocString(
    "This example program demonstrates solving the Navier-Stokes "
    "equation with Trilinos + MPI");


  bool write_vtu = true;
  clp.setOption("write-vtu",
                "no-vtu",
                &write_vtu,
                "Write results into vtu files?");
  // temporal finite element order
  int r = 0;
  clp.setOption("r", &r, "temporal FE degree");
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

  Step4 problem(r, write_vtu);
  problem.run();
}
