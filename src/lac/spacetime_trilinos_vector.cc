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

#include <ideal.II/lac/spacetime_trilinos_vector.hh>

#ifdef DEAL_II_WITH_TRILINOS
#  ifdef DEAL_II_WITH_MPI
namespace idealii::spacetime
{
  TrilinosVector::TrilinosVector()
  {
    _vectors = std::list<dealii::TrilinosWrappers::MPI::Vector>();
  }

  unsigned int
  TrilinosVector::M()
  {
    return _vectors.size();
  }

  void
  TrilinosVector::reinit(unsigned int M)
  {
    this->_vectors.clear();
    for (unsigned int i = 0; i < M; i++)
      {
        this->_vectors.push_back(dealii::TrilinosWrappers::MPI::Vector());
      }
  }

  slab::TrilinosVectorIterator
  TrilinosVector::begin()
  {
    return _vectors.begin();
  }

  slab::TrilinosVectorIterator
  TrilinosVector::end()
  {
    return _vectors.end();
  }
} // namespace idealii::spacetime

#  endif
#endif
