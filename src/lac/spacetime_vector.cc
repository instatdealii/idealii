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



#include <ideal.II/lac/spacetime_vector.hh>

namespace idealii{
namespace spacetime{
	template <typename Number>
	Vector<Number>::
	Vector(){
	  _vectors = std::list<dealii::Vector<Number>>();
	}

	template <typename Number>
	unsigned int
	Vector<Number>::M(){
		return _vectors.size();
	}

	template <typename Number>
	void
	Vector<Number>::reinit(unsigned int M){
	  this->_vectors.clear();
	  for (unsigned int i=0 ; i < M ; i++){
		  this->_vectors.push_back(dealii::Vector<Number>());
	  }
	}

	template <typename Number>
	slab::VectorIterator<Number>
	Vector<Number>::begin(){
		return _vectors.begin();
	}

	template <typename Number>
	slab::VectorIterator<Number>
	Vector<Number>::end(){
		return _vectors.end();
	}
}}

#include "spacetime_vector.inst"



