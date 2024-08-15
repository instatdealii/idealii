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
// same as in step-1
#include <ideal.II/base/quadrature_lib.hh>
#include <ideal.II/base/time_iterator.hh>

#include <ideal.II/dofs/slab_dof_tools.hh>
#include <ideal.II/dofs/spacetime_dof_handler.hh>

#include <ideal.II/fe/fe_dg.hh>
#include <ideal.II/fe/spacetime_fe_values.hh>

#include <ideal.II/grid/fixed_tria.hh>

#include <ideal.II/lac/spacetime_vector.hh>

#include <ideal.II/numerics/vector_tools.hh>

////////////////////////////////////////////
// @<H3> deal.II includes
////////////////////////////////////////////
// Same as in step-1
#include <deal.II/base/function.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

// For extracting the number of velocity and pressure dofs
#include <deal.II/dofs/dof_renumbering.h>

// For Stokes we have a system of finite elements
#include <deal.II/fe/fe_system.h>

// For reading grid description files
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

// Needed to ensure the circle obstacle is refined correctly
#include <deal.II/grid/manifold_lib.h>

////////////////////////////////////////////
// @<H3> C++ includes
////////////////////////////////////////////
//
#include <fstream>


////////////////////////////////////////////
// @<H2> Space-time functions
////////////////////////////////////////////

// This function describes the Dirichlet data for the inflow boundary,
// i.e. a Poiseuille flow or quadratic inflow profile.
// Note the use of get_time to obtain t.
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
         dealii::ExcIndexRange(component, 0, this->n_components));

  if (component == 0)
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

// The exact solution is unknown, so we can not specify it here.
// We don't need to specify the rhs function as it is zeros for this
// configuration.

////////////////////////////////////////////
// @<H2> The Step2 class
////////////////////////////////////////////
// This class describes the solution of Stokes equations with
// space-time slab tensor-product elements
class Step2
{
public:
  Step2(unsigned int temporal_degree = 1);
  void
  run();

private: // Nothing new here
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
  output_results_on_slab();

  idealii::spacetime::fixed::Triangulation<2> triangulation;
  idealii::spacetime::DoFHandler<2>           dof_handler;
  idealii::spacetime::Vector<double>          solution;

  idealii::spacetime::DG_FiniteElement<2> fe;

  dealii::SparsityPattern                            sparsity_pattern;
  std::shared_ptr<dealii::AffineConstraints<double>> slab_constraints;
  dealii::SparseMatrix<double>                       slab_system_matrix;
  dealii::Vector<double>                             slab_system_rhs;
  dealii::Vector<double>                             slab_initial_value;
  unsigned int                                       slab;

  struct
  {
    idealii::slab::TriaIterator<2>        tria;
    idealii::slab::DoFHandlerIterator<2>  dof;
    idealii::slab::VectorIterator<double> solution;
  } slab_its;
};



////////////////////////////////////////////
// @<H3> Step2::Step2
////////////////////////////////////////////
// Here, the constructor only takes the temporal degree as argument.
// The main change to step-1 is the construction of the finite element ``fe``,
// which is a continuous Taylor-Hood Q2/Q1 element in space consisting of
// * a vector-valued biquadratic element, i.e. Q2, for the velocity,
// * a scalar-valued bilinear element, i.e. Q1, for the pressure.
//
Step2::Step2(unsigned int temporal_degree)
  : triangulation()
  , dof_handler(&triangulation)
  , fe(std::make_shared<dealii::FESystem<2>>(dealii::FE_Q<2>(2),
                                             2,
                                             dealii::FE_Q<2>(1),
                                             1),
       temporal_degree)
  , slab(0)
{}

////////////////////////////////////////////
// @<H3> Step2::run
////////////////////////////////////////////
//
void
Step2::run() // same as Step1
{
  make_grid();
  time_marching();
}

////////////////////////////////////////////
// @<H3> Step2::make_grid
////////////////////////////////////////////
// In comparison to step-1 this function will read in a grid from file instead
// of using a generator.
void
Step2::make_grid()
{
  auto              space_tria = std::make_shared<dealii::Triangulation<2>>();
  dealii::GridIn<2> grid_in; // For reading in a mesh from file
  grid_in.attach_triangulation(*space_tria);
  std::ifstream input_file("nsbench4.inp");
  grid_in.read_ucd(input_file);
  // The interior obstacle is supposed to be a circle.
  // However, the input mesh specifies a polyhedron.
  // To ensure correct refinement we have to tell the triangulation
  // that this boundary is actually a circle.
  dealii::Point<2>                          p(0.2, 0.2);
  static const dealii::SphericalManifold<2> boundary(p);
  dealii::GridTools::copy_boundary_to_manifold_id(*space_tria);
  space_tria->set_manifold(80, boundary);

  const unsigned int M = 64;
  triangulation.generate(space_tria, M);
  triangulation.refine_global(2, 0);
  dof_handler.generate();
}

////////////////////////////////////////////
// @<H3> Step2::time_marching
////////////////////////////////////////////
//
void
Step2::time_marching() // same as in step-1
{
  idealii::TimeIteratorCollection<2> tic = idealii::TimeIteratorCollection<2>();

  solution.reinit(triangulation.M());

  slab_its.tria     = triangulation.begin();
  slab_its.dof      = dof_handler.begin();
  slab_its.solution = solution.begin();
  tic.add_iterator(&slab_its.tria, &triangulation);
  tic.add_iterator(&slab_its.dof, &dof_handler);
  tic.add_iterator(&slab_its.solution, &solution);

  std::cout << "*******Starting time-stepping*********" << std::endl;
  slab = 0;
  for (; !tic.at_end(); tic.increment())
    {
      std::cout << "Starting time-step (" << slab_its.tria->startpoint() << ","
                << slab_its.tria->endpoint() << "]" << std::endl;

      setup_system_on_slab();
      assemble_system_on_slab();
      solve_system_on_slab();
      output_results_on_slab();

      idealii::slab::VectorTools::extract_subvector_at_time_dof(
        *slab_its.solution,
        slab_initial_value,
        slab_its.tria->temporal()->n_global_active_cells() - 1);

      slab++;
    }
}

////////////////////////////////////////////
// @<H3> Step2::setup_system_on_slab
////////////////////////////////////////////
//
void
Step2::setup_system_on_slab()
{
  slab_its.dof->distribute_dofs(fe);
  // We reorder the DoFs such that all DoFs belonging to a specific
  // component are arranged together.
  dealii::DoFRenumbering::component_wise(*slab_its.dof->spatial());

  std::cout << "Number of degrees of freedom: \n\t"
            << slab_its.dof->n_dofs_space() << " (space) * "
            << slab_its.dof->n_dofs_time()
            << " (time) = " << slab_its.dof->n_dofs_spacetime()
            << " (spacetime)" << std::endl;

  if (slab == 0)
    {
      slab_initial_value.reinit(slab_its.dof->n_dofs_space());
      slab_initial_value = 0;
    }

  slab_constraints = std::make_shared<dealii::AffineConstraints<double>>();

  // Now we have multiple Dirichlet boundaries for the velocity.
  // The boundary ids are specified in the grid input file.
  //
  // * Inhomogenous Poisseuille inflow at the left wall (id 0)
  // * No-slip condition(:math:`v=0`) at both the upper and lower
  //   walls (id 2) as well as the obstacle (id 80).
  auto              zero   = dealii::Functions::ZeroFunction<2>(3);
  auto              inflow = PoisseuilleInflow();
  std::vector<bool> component_mask(3, true);
  component_mask[2] = false;

  idealii::slab::VectorTools::interpolate_boundary_values(
    *slab_its.dof, 0, inflow, slab_constraints, component_mask);

  idealii::slab::VectorTools::interpolate_boundary_values(
    *slab_its.dof, 2, zero, slab_constraints, component_mask);

  idealii::slab::VectorTools::interpolate_boundary_values(
    *slab_its.dof, 80, zero, slab_constraints, component_mask);

  slab_constraints->close();


  // Specify coupling between components to save on memory, since the pressure
  // test and trial functions are not multiplied in the equation.
  // Therefore, the sparsity can be zero in the pressure-pressure block.
  dealii::Table<2, dealii::DoFTools::Coupling> coupling_space(2 + 1, 2 + 1);
  coupling_space.fill(dealii::DoFTools::none);
  for (unsigned int i = 0; i < 2; i++)
    {
      coupling_space[i][2] = dealii::DoFTools::always; //(v,p)
      coupling_space[2][i] = dealii::DoFTools::always; //(p,v)
      for (unsigned int j = 0; j < 2; j++)
        {
          coupling_space[i][j] = dealii::DoFTools::always; //(v,v)
        }
    }

  // Now we construct a sparsity pattern with the given coupling and
  // constraints.
  // Apart from that, the rest of the function is the same as before.
  dealii::DynamicSparsityPattern dsp(slab_its.dof->n_dofs_spacetime());
  idealii::slab::DoFTools::make_upwind_sparsity_pattern(*slab_its.dof,
                                                        coupling_space,
                                                        dsp,
                                                        slab_constraints);
  sparsity_pattern.copy_from(dsp);
  slab_system_matrix.reinit(sparsity_pattern);

  slab_its.solution->reinit(slab_its.dof->n_dofs_spacetime());
  slab_system_rhs.reinit(slab_its.dof->n_dofs_spacetime());
}

////////////////////////////////////////////
// @<H3> Step2::assemble_system_on_slab
////////////////////////////////////////////
//
void
Step2::assemble_system_on_slab()
{
  // The beginning is the same as before.
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
  dealii::Vector<double>     cell_rhs(N * dofs_per_spacetime_cell);

  std::vector<dealii::types::global_dof_index> local_spacetime_dof_index(
    N * dofs_per_spacetime_cell);

  unsigned int n;
  unsigned int n_quad_spacetime = fe_values_spacetime.n_quadrature_points;
  unsigned int n_quad_space     = quad.spatial()->size();

  // Set the kinematic viscosity
  double nu_f = 1.0e-3;

  // Specify the type of unknown, i.e. Vector or Scalar and the beginning index
  // in the ``FESystem``.
  dealii::FEValuesExtractors::Vector velocity(0);
  dealii::FEValuesExtractors::Scalar pressure(2);


  // Same as before, we split the space-time element loop into spatial and
  // temporal components.
  for (const auto &cell_space :
       slab_its.dof->spatial()->active_cell_iterators())
    {
      fe_values_spacetime.reinit_space(cell_space);
      fe_jump_values_spacetime.reinit_space(cell_space);
      std::vector<dealii::Tensor<1, 2>> initial_values(
        fe_values_spacetime.spatial()->n_quadrature_points);
      // Get the function values of the velocity component of the spatial finite
      // element.
      fe_values_spacetime.spatial()->operator[](velocity).get_function_values(
        slab_initial_value, initial_values);

      cell_matrix = 0;
      cell_rhs    = 0;
      for (const auto &cell_time :
           slab_its.dof->temporal()->active_cell_iterators())
        {
          n = cell_time->index();
          fe_values_spacetime.reinit_time(cell_time);
          fe_jump_values_spacetime.reinit_time(cell_time);
          fe_values_spacetime.get_local_dof_indices(local_spacetime_dof_index);

          for (unsigned int q = 0; q < n_quad_spacetime; ++q)
            {
              for (unsigned int i = 0; i < dofs_per_spacetime_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_spacetime_cell; ++j)
                    {
                      // The FEValues calls with extractors are different
                      // to the ones in deal.II to avoid duplicate caching.
                      // Instead of ``operator[]`` returning a specialized
                      // ``FEValuesViews`` object, ``FEValues`` directly offers
                      // functions for scalar and vector valued shape functions.
                      //
                      // Example:
                      //
                      // * deal.II: ``fe_values[velocity].value(i,q)``
                      // * ideal.II: ``fe_values.vector_value(extractor,i,q)`

                      // :math:`(\partial_t u, v)`
                      cell_matrix(i + n * dofs_per_spacetime_cell,
                                  j + n * dofs_per_spacetime_cell) +=
                        fe_values_spacetime.vector_value(velocity, i, q) *
                        fe_values_spacetime.vector_dt(velocity, j, q) *
                        fe_values_spacetime.JxW(q);

                      // :math:`(\nabla u, \nabla v)`
                      // Since the shape function is vector valued,
                      // we have to call ``scalar_product()``
                      cell_matrix(i + n * dofs_per_spacetime_cell,
                                  j + n * dofs_per_spacetime_cell) +=
                        dealii::scalar_product(
                          fe_values_spacetime.vector_space_grad(velocity, i, q),
                          fe_values_spacetime.vector_space_grad(velocity,
                                                                j,
                                                                q)) *
                        fe_values_spacetime.JxW(q) * nu_f;

                      // :math:`-(p, \nabla\cdot v)`
                      cell_matrix(i + n * dofs_per_spacetime_cell,
                                  j + n * dofs_per_spacetime_cell) -=
                        fe_values_spacetime.vector_divergence(velocity, i, q) *
                        fe_values_spacetime.scalar_value(pressure, j, q) *
                        fe_values_spacetime.JxW(q);

                      // :math:`(\nabla\cdot u,q)` (div free constraint)
                      cell_matrix(i + n * dofs_per_spacetime_cell,
                                  j + n * dofs_per_spacetime_cell) +=
                        fe_values_spacetime.scalar_value(pressure, i, q) *
                        fe_values_spacetime.vector_divergence(velocity, j, q) *
                        fe_values_spacetime.JxW(q);

                    } // dofs j

                } // dofs i

            } // quad

          // Only the velocity has a temporal derivative, so we don't need
          // jump values for the pressure.
          for (unsigned int q = 0; q < n_quad_space; ++q)
            {
              for (unsigned int i = 0; i < dofs_per_spacetime_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_spacetime_cell; ++j)
                    {
                      // :math:`(u^+, v^+)`
                      cell_matrix(i + n * dofs_per_spacetime_cell,
                                  j + n * dofs_per_spacetime_cell) +=
                        fe_jump_values_spacetime.vector_value_plus(velocity,
                                                                   i,
                                                                   q) *
                        fe_jump_values_spacetime.vector_value_plus(velocity,
                                                                   j,
                                                                   q) *
                        fe_jump_values_spacetime.JxW(q);

                      // :math:`-(u^-, v^+)`
                      if (n > 0)
                        {
                          cell_matrix(i + n * dofs_per_spacetime_cell,
                                      j + (n - 1) * dofs_per_spacetime_cell) -=
                            fe_jump_values_spacetime.vector_value_plus(velocity,
                                                                       i,
                                                                       q) *
                            fe_jump_values_spacetime.vector_value_minus(
                              velocity, j, q) *
                            fe_jump_values_spacetime.JxW(q);
                        }
                    } // dofs j

                  if (n == 0)
                    {
                      cell_rhs(i) +=
                        fe_jump_values_spacetime.vector_value_plus(velocity,
                                                                   i,
                                                                   q) *
                        initial_values[q] // value of previous solution at t0
                        * fe_jump_values_spacetime.JxW(q);
                    }
                } // dofs i
            }

        } // cell time
      slab_constraints->distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_spacetime_dof_index,
                                                   slab_system_matrix,
                                                   slab_system_rhs);
    } // cell space
}

////////////////////////////////////////////
// @<H3> Step2::solve_system_on_slab
////////////////////////////////////////////
//
void
Step2::solve_system_on_slab() // same as before
{
  dealii::SparseDirectUMFPACK solver;
  solver.factorize(slab_system_matrix);
  solver.vmult(*slab_its.solution, slab_system_rhs);
  slab_constraints->distribute(*slab_its.solution);
}

////////////////////////////////////////////
// @<H3> Step2::output_results_on_slab
////////////////////////////////////////////
//
void
Step2::output_results_on_slab()
{
  auto n_dofs = slab_its.dof->n_dofs_time();
  // As in deal.II we need to tell the DataOut what to do with the vector and
  // scalar valued entries in the solution vector.
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

  for (unsigned i = 0; i < n_dofs; i++)
    {
      dealii::DataOut<2> data_out;
      data_out.attach_dof_handler(*slab_its.dof->spatial());
      dealii::Vector<double> local_solution;
      local_solution.reinit(slab_its.dof->n_dofs_space());
      idealii::slab::VectorTools::extract_subvector_at_time_dof(
        *slab_its.solution, local_solution, i);
      data_out.add_data_vector(local_solution,
                               field_names,
                               dealii::DataOut<2>::type_dof_data,
                               dci);

      data_out.build_patches(1);
      std::ostringstream filename;
      filename << "solution_dG(" << fe.temporal()->degree << ")_t_"
               << slab * n_dofs + i << ".vtk";

      std::ofstream output(filename.str());
      data_out.write_vtk(output);
      output.close();
    }
}

////////////////////////////////////////////
// @<H2> The main function
////////////////////////////////////////////
//
int
main()
{
  Step2 problem(0);
  problem.run();
}
