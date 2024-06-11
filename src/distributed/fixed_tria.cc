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

#include <ideal.II/distributed/fixed_tria.hh>

#ifdef DEAL_II_WITH_MPI
namespace idealii::spacetime::parallel::distributed::fixed
{
  template <int dim>
  Triangulation<dim>::Triangulation(
    dealii::types::global_cell_index max_N_intervals_per_slab)
    : spacetime::parallel::distributed::Triangulation<dim>(
        max_N_intervals_per_slab)
  {}

  template <int dim>
  void
  Triangulation<dim>::generate(
    std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>>
                 space_tria,
    unsigned int M,
    double       t0,
    double       T)
  {
    Assert(space_tria.use_count(), dealii::ExcNotInitialized());
    // Todo: somehow assert that space_tria is actually parallel distributed
    double t = t0;
    double k = (T - t0) / M;
    for (unsigned int i = 0; i < M; i++)
      {
        this->trias.push_back(
          idealii::slab::parallel::distributed::Triangulation<dim>(space_tria,
                                                                   t,
                                                                   t + k));
        t += k;
      }
  }

  template <int dim>
  void
  Triangulation<dim>::refine_global(const unsigned int times_space,
                                    const unsigned int times_time)
  {
    slab::parallel::distributed::TriaIterator<dim> slab_tria = this->begin();
    slab_tria->spatial()->refine_global(times_space);
    for (; slab_tria != this->end(); ++slab_tria)
      {
        slab_tria->temporal()->refine_global(times_time);


        // Check if temporal triangulation got too large (unless max_N = 0)
        if (this->max_N_intervals_per_slab &&
            slab_tria->temporal()->n_global_active_cells() >
              this->max_N_intervals_per_slab)
          {
            dealii::types::global_cell_index M =
              slab_tria->temporal()->n_global_active_cells();
            std::vector<double>              step_sizes(M);
            dealii::types::global_cell_index i = 0;
            for (auto &cell : slab_tria->temporal()->active_cell_iterators())
              {
                step_sizes[i] = cell->bounding_box().side_length(0);
                i++;
              }

            // number of subdivisions needed is at least
            dealii::types::global_cell_index subdiv =
              M / this->max_N_intervals_per_slab;
            // remaining intervals?
            dealii::types::global_cell_index modulus =
              M % this->max_N_intervals_per_slab;

            std::vector<std::vector<double>> partial_step_sizes;
            for (dealii::types::global_cell_index j = 0; j < subdiv; j++)
              {
                partial_step_sizes.push_back(
                  std::vector<double>(this->max_N_intervals_per_slab));
              }
            if (modulus)
              {
                partial_step_sizes.push_back(std::vector<double>(modulus));
              }


            // Distribute step sizes onto subdivisions
            for (i = 0; i < M; ++i)
              {
                dealii::types::global_cell_index j =
                  i / this->max_N_intervals_per_slab;
                dealii::types::global_cell_index k =
                  i % this->max_N_intervals_per_slab;
                partial_step_sizes[j][k] = step_sizes[i];
              }

            // we need to calculate temporal bounds for the new slabs
            double end   = slab_tria->startpoint();
            double start = end;

            // The first new slabs will be emplaced before the current one.
            // Then, they will not invalidate the list iterator going forward
            for (i = 0; i < partial_step_sizes.size() - 1; ++i)
              {
                start = end;
                for (dealii::types::global_cell_index j = 0;
                     j < partial_step_sizes[i].size();
                     ++j)
                  {
                    end += partial_step_sizes[i][j];
                  }
                this->trias.emplace(slab_tria,
                                    slab_tria->spatial(),
                                    partial_step_sizes[i],
                                    start,
                                    end);
              }
            // Finally, update the current slab triangulation
            start = end;
            for (dealii::types::global_cell_index j = 0;
                 j < partial_step_sizes[i].size();
                 ++j)
              {
                end += partial_step_sizes[i][j];
              }
            slab_tria->update_temporal_triangulation(partial_step_sizes[i],
                                                     start,
                                                     end);
          }
      }
  }
} // namespace idealii::spacetime::parallel::distributed::fixed

#  include "fixed_tria.inst"

#endif
