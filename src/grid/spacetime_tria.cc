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


#include <ideal.II/grid/spacetime_tria.hh>

namespace idealii{
namespace spacetime{
	template<int dim>
	Triangulation<dim>::
	Triangulation(){
		trias=std::list<idealii::slab::Triangulation<dim>>();
	}

	template<int dim>
	unsigned int
	Triangulation<dim>::M(){
		return trias.size();
	}

	template<int dim>
	slab::TriaIterator<dim>
	Triangulation<dim>::begin(){
		return trias.begin();
	}

	template<int dim>
	slab::TriaIterator<dim>
	Triangulation<dim>::end(){
		return trias.end();
	}

}}

#include "spacetime_tria.inst"
