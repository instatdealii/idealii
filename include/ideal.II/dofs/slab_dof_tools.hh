/*
 * slab_dof_tools.hh
 *
 *  Created on: Nov 22, 2022
 *      Author: thiele
 */

#ifndef INCLUDE_IDEAL_II_DOFS_SLAB_DOF_TOOLS_HH_
#define INCLUDE_IDEAL_II_DOFS_SLAB_DOF_TOOLS_HH_

#include <ideal.II/dofs/slab_dof_handler.hh>

#include <ideal.II/fe/fe_dg.hh>

#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace idealii::slab::DoFTools
{
  namespace internal
  {
    template <int dim>
    void
    upwind_temporal_pattern(DoFHandler<dim>                &dof,
                            dealii::DynamicSparsityPattern &time_dsp)
    {
      dealii::DoFTools::make_sparsity_pattern(*dof.temporal(), time_dsp);
      if (dof.fe_support_type() ==
          spacetime::DG_FiniteElement<dim>::support_type::Lobatto)
        {
          for (dealii::types::global_dof_index ii = dof.dofs_per_cell_time();
               ii < dof.n_dofs_time();
               ii += dof.dofs_per_cell_time())
            {
              time_dsp.add(ii, ii - 1);
            }
        }
      else if (dof.fe_support_type() ==
               spacetime::DG_FiniteElement<dim>::support_type::RadauLeft)
        {
          for (dealii::types::global_dof_index ii = dof.dofs_per_cell_time();
               ii < dof.n_dofs_time();
               ii += dof.dofs_per_cell_time())
            {
              for (dealii::types::global_dof_index l = 0;
                   l < dof.dofs_per_cell_time();
                   l++)
                {
                  time_dsp.add(ii, ii - l - 1);
                }
            }
        }
      else if (dof.fe_support_type() ==
               spacetime::DG_FiniteElement<dim>::support_type::RadauRight)
        {
          for (dealii::types::global_dof_index ii = dof.dofs_per_cell_time();
               ii < dof.n_dofs_time();
               ii += dof.dofs_per_cell_time())
            {
              for (dealii::types::global_dof_index k = 0;
                   k < dof.dofs_per_cell_time();
                   k++)
                {
                  time_dsp.add(ii + k, ii - 1);
                }
            }
        }
      else
        {
          // go over first DoF of each cell
          for (dealii::types::global_dof_index ii = dof.dofs_per_cell_time();
               ii < dof.n_dofs_time();
               ii += dof.dofs_per_cell_time())
            {
              // row offset
              for (dealii::types::global_dof_index k = 0;
                   k < dof.dofs_per_cell_time();
                   k++)
                {
                  for (dealii::types::global_dof_index l = 0;
                       l < dof.dofs_per_cell_time();
                       l++)
                    {
                      time_dsp.add(ii + k, ii - l - 1);
                    }
                }
            }
        }
    }
    template <int dim>
    void
    downwind_temporal_pattern(DoFHandler<dim>                &dof,
                              dealii::DynamicSparsityPattern &time_dsp)
    {
      dealii::DoFTools::make_sparsity_pattern(*dof.temporal(), time_dsp);
      if (dof.fe_support_type() ==
          spacetime::DG_FiniteElement<dim>::support_type::Lobatto)
        {
          for (dealii::types::global_dof_index ii = dof.dofs_per_cell_time();
               ii < dof.n_dofs_time();
               ii += dof.dofs_per_cell_time())
            {
              time_dsp.add(ii - 1, ii);
            }
        }
      else if (dof.fe_support_type() ==
               spacetime::DG_FiniteElement<dim>::support_type::RadauRight)
        {
          for (dealii::types::global_dof_index ii = dof.dofs_per_cell_time();
               ii < dof.n_dofs_time();
               ii += dof.dofs_per_cell_time())
            {
              for (dealii::types::global_dof_index k = 0;
                   k < dof.dofs_per_cell_time();
                   k++)
                {
                  time_dsp.add(ii - 1, ii + k);
                }
            }
        }
      else if (dof.fe_support_type() ==
               spacetime::DG_FiniteElement<dim>::support_type::RadauLeft)
        {
          for (dealii::types::global_dof_index ii = dof.dofs_per_cell_time();
               ii < dof.n_dofs_time();
               ii += dof.dofs_per_cell_time())
            {
              for (dealii::types::global_dof_index l = 0;
                   l < dof.dofs_per_cell_time();
                   l++)
                {
                  time_dsp.add(ii - l - 1, ii);
                }
            }
        }
      else
        {
          // go over first DoF of each cell
          for (dealii::types::global_dof_index ii = dof.dofs_per_cell_time();
               ii < dof.n_dofs_time();
               ii += dof.dofs_per_cell_time())
            {
              // row offset
              for (dealii::types::global_dof_index k = 0;
                   k < dof.dofs_per_cell_time();
                   k++)
                {
                  for (dealii::types::global_dof_index l = 0;
                       l < dof.dofs_per_cell_time();
                       l++)
                    {
                      time_dsp.add(ii - l - 1, ii + k);
                    }
                }
            }
        }
    }
  } // namespace internal

  /**
   * @brief Construction of a sparsity pattern with lower off diagonal jump terms.
   *
   * This functions constructs the tensor product between
   * a spatial sparsity pattern and a temporal upwind sparsity.
   *
   * The spatial pattern is constructed by calling the function
   * make_sparsity_pattern()
   * with the spatial DoFHandler of @p dof, the @p space_contraints and @p keep_constrained_dofs.
   *
   * The temporal pattern is initially constructed as a block-diagonal sparsity
   * pattern
   * by the corresponding deal.II method for the temporal DoFHandler of @p dof.
   * If the number of temporal elements in this slab is greater than one,
   * additional off diagonal entries for the jump terms between a temporal
   * element and its direct left/earlier neighbor are added.
   *
   * For Gauss-Lobatto support points in time this is only a coupling between
   * the initial temporal dof of the current element and
   * the final temporal dof of the neighbor.
   *
   * For Gauss-Legendre support points in time this is the complete first lower
   * off diagonal block.
   *
   * @param dof A shared pointer to the DoFHandler object to base the patterns on.
   * @param st_dsp The dynamic sparsity pattern that will afterwards include the spacetime pattern.
   * @param space_constraints The spatial constraints handed to the spatial pattern function
   * @param keep_constrained_dofs When using distribute_local_to_global() of the spacetime constraints
   * off diagonal entries of constrained dofs will not be written. So it is
   * possible to not include these in the sparsity pattern. For more details see
   * the functions in the dealii::DoFTools namespace.
   */
  template <int dim>
  void
  make_upwind_sparsity_pattern(
    DoFHandler<dim>                                   &dof,
    dealii::DynamicSparsityPattern                    &st_dsp,
    std::shared_ptr<dealii::AffineConstraints<double>> space_constraints =
      std::make_shared<dealii::AffineConstraints<double>>(),
    const bool keep_constrained_dofs = true)
  {
    dealii::DynamicSparsityPattern space_dsp(dof.n_dofs_space());
    dealii::DoFTools::make_sparsity_pattern(*dof.spatial(),
                                            space_dsp,
                                            *space_constraints,
                                            keep_constrained_dofs);

    dealii::DynamicSparsityPattern time_dsp(dof.n_dofs_time());
    internal::upwind_temporal_pattern(dof, time_dsp);

    for (auto &space_entry : space_dsp)
      {
        for (auto &time_entry : time_dsp)
          {
            st_dsp.add(time_entry.row() * dof.n_dofs_space() +
                         space_entry.row(), // test function
                       time_entry.column() * dof.n_dofs_space() +
                         space_entry.column() // trial function
            );
          }
      }
  }
  /**
   * @brief Construction of a sparsity pattern with lower off diagonal jump terms.
   *
   * This functions works like the previous one butthe spatial pattern is
   * constructed using the supplied couplings.
   * @param space_couplings In coupled problems some components
   * don't couple in the weak formulation and consequently don't needs
   * entries in the sparsity pattern.
   *
   * An example would be the Stokes system where pressure ansatz function is not
   * multiplied by the pressure test function, which leads to an empty diagonal
   * block.
   */
  template <int dim>
  void
  make_upwind_sparsity_pattern(
    DoFHandler<dim>                                    &dof,
    const dealii::Table<2, dealii::DoFTools::Coupling> &space_couplings,
    dealii::DynamicSparsityPattern                     &st_dsp,
    std::shared_ptr<dealii::AffineConstraints<double>>  space_constraints =
      std::make_shared<dealii::AffineConstraints<double>>(),
    const bool keep_constrained_dofs = true)
  {
    dealii::DynamicSparsityPattern space_dsp(dof.n_dofs_space());
    dealii::DoFTools::make_sparsity_pattern(*dof.spatial(),
                                            space_couplings,
                                            space_dsp,
                                            *space_constraints,
                                            keep_constrained_dofs);

    dealii::DynamicSparsityPattern time_dsp(dof.n_dofs_time());
    internal::upwind_temporal_pattern(dof, time_dsp);

    for (auto &space_entry : space_dsp)
      {
        for (auto &time_entry : time_dsp)
          {
            st_dsp.add(time_entry.row() * dof.n_dofs_space() +
                         space_entry.row(), // test function
                       time_entry.column() * dof.n_dofs_space() +
                         space_entry.column() // trial function
            );
          }
      }
  }

  /**
   * @brief Construction of a sparsity pattern with upper off diagonal jump terms.
   *
   * This functions constructs the tensor product between
   * a spatial sparsity pattern and a temporal upwind sparsity.
   *
   * The spatial pattern is constructed by calling the function
   * make_sparsity_pattern()
   * with the spatial DoFHandler of @p dof, the @p space_contraints and @p keep_constrained_dofs.
   *
   * The temporal pattern is initially constructed as a block-diagonal sparsity
   * pattern
   * by the corresponding deal.II method for the temporal DoFHandler of @p dof.
   * If the number of temporal elements in this slab is greater than one,
   * additional off diagonal entries for the jump terms between a temporal
   * element and its direct right/later neighbor are added.
   *
   * For Gauss-Lobatto support points in time this is only a coupling between
   * the final temporal dof of the current element and
   * the initial temporal dof of the neighbor.
   *
   * For Gauss-Legendre support points in time this is the complete first upper
   * off diagonal block.
   *
   * @param dof A shared pointer to the DoFHandler object to base the patterns on.
   * @param st_dsp The dynamic sparsity pattern that will afterwards include the spacetime pattern.
   * @param space_constraints The spatial constraints handed to the spatial pattern function
   * @param keep_constrained_dofs When using distribute_local_to_global() of the spacetime constraints
   * off diagonal entries of constrained dofs will not be written. So it is
   * possible to not include these in the sparsity pattern. For more details see
   * the functions in the dealii::DoFTools namespace.
   */
  template <int dim>
  void
  make_downwind_sparsity_pattern(
    DoFHandler<dim>                                   &dof,
    dealii::DynamicSparsityPattern                    &st_dsp,
    std::shared_ptr<dealii::AffineConstraints<double>> space_constraints =
      std::make_shared<dealii::AffineConstraints<double>>(),
    const bool keep_constrained_dofs = true)
  {
    dealii::DynamicSparsityPattern space_dsp(dof.n_dofs_space());
    dealii::DoFTools::make_sparsity_pattern(*dof.spatial(),
                                            space_dsp,
                                            *space_constraints,
                                            keep_constrained_dofs);

    dealii::DynamicSparsityPattern time_dsp(dof.n_dofs_time());
    internal::downwind_temporal_pattern(dof, time_dsp);

    for (auto &space_entry : space_dsp)
      {
        for (auto &time_entry : time_dsp)
          {
            st_dsp.add(time_entry.row() * dof.n_dofs_space() +
                         space_entry.row(), // test function
                       time_entry.column() * dof.n_dofs_space() +
                         space_entry.column() // trial function
            );
          }
      }
  }
  /**
   * @brief Construction of a sparsity pattern with upper off diagonal jump terms.
   *
   * This functions works like the previous one but the spatial pattern is
   * constructed using the supplied couplings.
   * @param space_couplings In coupled problems some components
   * don't couple in the weak formulation and consequently don't needs
   * entries in the sparsity pattern.
   *
   * An example would be the Stokes system where pressure ansatz function is not
   * multiplied by the pressure test function, which leads to an empty diagonal
   * block.
   */
  template <int dim>
  void
  make_downwind_sparsity_pattern(
    DoFHandler<dim>                                    &dof,
    const dealii::Table<2, dealii::DoFTools::Coupling> &space_couplings,
    dealii::DynamicSparsityPattern                     &st_dsp,
    std::shared_ptr<dealii::AffineConstraints<double>>  space_constraints =
      std::make_shared<dealii::AffineConstraints<double>>(),
    const bool keep_constrained_dofs = true)
  {
    dealii::DynamicSparsityPattern space_dsp(dof.n_dofs_space());
    dealii::DoFTools::make_sparsity_pattern(*dof.spatial(),
                                            space_couplings,
                                            space_dsp,
                                            *space_constraints,
                                            keep_constrained_dofs);

    dealii::DynamicSparsityPattern time_dsp(dof.n_dofs_time());
    internal::downwind_temporal_pattern(dof, time_dsp);

    for (auto &space_entry : space_dsp)
      {
        for (auto &time_entry : time_dsp)
          {
            st_dsp.add(time_entry.row() * dof.n_dofs_space() +
                         space_entry.row(), // test function
                       time_entry.column() * dof.n_dofs_space() +
                         space_entry.column() // trial function
            );
          }
      }
  }

  /**
   * @brief Construction of space-time hanging node constraints.
   *
   * This function internally constructs spatial hanging node constraints based
   * on the spatial part of @dof_handler.
   *
   * The spacetime constraints are then obtained by offsetting the constraints
   * by the total number of dofs in the spatial dof handler.
   *
   * @param dof_handler
   * @param spacetime_constraints
   */
  template <int dim, typename Number>
  void
  make_hanging_node_constraints(
    idealii::slab::DoFHandler<dim>                    &dof_handler,
    std::shared_ptr<dealii::AffineConstraints<Number>> spacetime_constraints)
  {
    auto space_constraints =
      std::make_shared<dealii::AffineConstraints<Number>>();
    dealii::DoFTools::make_hanging_node_constraints(*dof_handler.spatial(),
                                                    *space_constraints);

    unsigned int n_space_dofs = dof_handler.n_dofs_space();
    for (unsigned int time_dof = 0; time_dof < dof_handler.n_dofs_time();
         time_dof++)
      {
        for (unsigned int i = 0; i < n_space_dofs; i++)
          {
            if (space_constraints->is_constrained(i))
              {
                const std::vector<
                  std::pair<dealii::types::global_dof_index, double>> *entries =
                  space_constraints->get_constraint_entries(i);
                spacetime_constraints->add_line(i + time_dof * n_space_dofs);
                // non Dirichlet constraint
                if (entries->size() > 0)
                  {
                    for (auto entry : *entries)
                      {
                        spacetime_constraints->add_entry(
                          i + time_dof * n_space_dofs,
                          entry.first + time_dof * n_space_dofs,
                          entry.second);
                      }
                  }
                else
                  {
                    spacetime_constraints->set_inhomogeneity(
                      i + time_dof * n_space_dofs,
                      space_constraints->get_inhomogeneity(i));
                  }
              }
          }
      }
  }

  /**
   * @brief Construction of parallel distributed space-time hanging node constraints.
   *
   * This function internally constructs spatial hanging node constraints based
   * on the spatial part of @dof_handler.
   *
   * The spacetime constraints are then obtained by offsetting the constraints
   * by the total number of dofs in the spatial dof handler.
   *
   * @param dof_handler
   * @param spacetime_constraints
   */
  template <int dim, typename Number>
  void
  make_hanging_node_constraints(
    dealii::IndexSet                                  &space_relevant_dofs,
    idealii::slab::DoFHandler<dim>                    &dof_handler,
    std::shared_ptr<dealii::AffineConstraints<Number>> spacetime_constraints)
  {
    auto space_constraints =
      std::make_shared<dealii::AffineConstraints<Number>>();
    space_constraints->reinit(space_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(*dof_handler.spatial(),
                                                    *space_constraints);

    unsigned int n_space_dofs = dof_handler.n_dofs_space();
    for (unsigned int time_dof = 0; time_dof < dof_handler.n_dofs_time();
         time_dof++)
      {
        for (auto id = space_relevant_dofs.begin();
             id != space_relevant_dofs.end();
             id++)
          {
            if (space_constraints->is_constrained(*id))
              {
                const std::vector<
                  std::pair<dealii::types::global_dof_index, double>> *entries =
                  space_constraints->get_constraint_entries(*id);

                spacetime_constraints->add_line(*id + time_dof * n_space_dofs);
                // non Dirichlet constraint
                if (entries->size() > 0)
                  {
                    for (auto entry : *entries)
                      {
                        spacetime_constraints->add_entry(
                          *id + time_dof * n_space_dofs,
                          entry.first + time_dof * n_space_dofs,
                          entry.second);
                      }
                  }
                else
                  {
                    spacetime_constraints->set_inhomogeneity(
                      *id + time_dof * n_space_dofs,
                      space_constraints->get_inhomogeneity(*id));
                  }
              }
          }
      }
  }

  /**
   * @brief Extract global space-time dof indices active on the current DoFHandler.
   *
   * For DoFHandlers with a parallel triangulation this function returns the
   * locally owned dofs and all dofs belonging to other processors that are
   * located at a locally owned cell.s
   */
  template <int dim>
  dealii::IndexSet
  extract_locally_relevant_dofs(slab::DoFHandler<dim> &dof_handler)
  {
    dealii::IndexSet space_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler.spatial(),
                                                    space_relevant_dofs);

    dealii::IndexSet spacetime_relevant_dofs(space_relevant_dofs.size() *
                                             dof_handler.n_dofs_time());

    for (dealii::types::global_dof_index time_dof_index = 0;
         time_dof_index < dof_handler.n_dofs_time();
         time_dof_index++)
      {
        spacetime_relevant_dofs.add_indices(
          space_relevant_dofs,
          time_dof_index * dof_handler.n_dofs_space() // offset
        );
      }

    return spacetime_relevant_dofs;
  }

} // namespace idealii::slab::DoFTools

#endif /* INCLUDE_IDEAL_II_DOFS_SLAB_DOF_TOOLS_HH_ */
