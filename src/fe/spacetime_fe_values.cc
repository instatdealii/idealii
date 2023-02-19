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


#include <ideal.II/fe/spacetime_fe_values.hh>

namespace idealii{
namespace spacetime{


 ////////////////////////////////////////////////////////////
 // FEValues ////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////
	template <int dim>
	FEValues<dim>::FEValues(DG_FiniteElement<dim>& fe,
			                Quadrature<dim>& quad,
							const dealii::UpdateFlags uflags)
	: _fe(fe),
      _quad(quad),
	  _fev_space(std::make_shared<dealii::FEValues<dim>>(*fe.spatial(),*quad.spatial(),uflags)),
	  _fev_time(std::make_shared<dealii::FEValues<1>>(*fe.temporal(),*quad.temporal(),uflags)),
	  local_space_dof_index(fe.spatial()->dofs_per_cell),
	  local_time_dof_index(fe.temporal()->dofs_per_cell),
	  n_dofs_space(0),
	  time_cell_index(0),
	  n_dofs_space_cell(_fe.spatial()->dofs_per_cell),
	  n_quads_space(_fev_space->n_quadrature_points),
	  n_quadrature_points(_fev_space->n_quadrature_points*_fev_time->n_quadrature_points){}

	template <int dim>
	void
	FEValues<dim>::reinit_space(const typename dealii::TriaIterator<dealii::DoFCellAccessor<dim,dim,false>>& cell_space){
		_fev_space->reinit(cell_space);
		cell_space->get_dof_indices(local_space_dof_index);
		n_dofs_space = cell_space->get_dof_handler().n_dofs();
	}

	template <int dim>
	void
	FEValues<dim>::reinit_time(const typename dealii::TriaIterator<dealii::DoFCellAccessor<1,1,false>>& cell_time){
		_fev_time->reinit(cell_time);
		cell_time->get_dof_indices(local_time_dof_index);
		time_cell_index = cell_time->index();
	}

	template <int dim>
	double
	FEValues<dim>::shape_value(unsigned int function_no,
							   unsigned int point_no){
		return
			_fev_space->shape_value(function_no%n_dofs_space_cell,
					                point_no%n_quads_space)*
			_fev_time->shape_value(function_no/n_dofs_space_cell,
			   	                   point_no/n_quads_space);
	}

	template <int dim>
	double
	FEValues<dim>::shape_dt(unsigned int function_no,
							   unsigned int point_no){
		return
			_fev_space->shape_value(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_grad(function_no/n_dofs_space_cell,
								   point_no/n_quads_space)[0];
	}

	template <int dim>
	dealii::Tensor< 1, dim >
	FEValues<dim>::shape_space_grad(unsigned int function_no,
							   unsigned int point_no){
		return
			_fev_space->shape_grad(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_value(function_no/n_dofs_space_cell,
								   point_no/n_quads_space);
	}

	template <int dim>
	typename dealii::FEValuesViews::Scalar<dim>::value_type
	FEValues<dim>::scalar_value(const typename dealii::FEValuesExtractors::Scalar& extractor,
								unsigned int function_no,
								unsigned int point_no){
		return
			(*_fev_space)[extractor].value(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_value(function_no/n_dofs_space_cell,
								   point_no/n_quads_space);
	}

	template <int dim>
	typename dealii::FEValuesViews::Scalar<dim>::value_type
	FEValues<dim>::scalar_dt(const typename dealii::FEValuesExtractors::Scalar& extractor,
							 unsigned int function_no,
							 unsigned int point_no){
		return
			(*_fev_space)[extractor].value(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_grad(function_no/n_dofs_space_cell,
								   point_no/n_quads_space)[0];
	}

	template <int dim>
	typename dealii::FEValuesViews::Scalar<dim>::gradient_type
	FEValues<dim>::scalar_space_grad(const typename dealii::FEValuesExtractors::Scalar& extractor,
									 unsigned int function_no,
									 unsigned int point_no){
		return
			(*_fev_space)[extractor].gradient(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_value(function_no/n_dofs_space_cell,
								   point_no/n_quads_space);
	}
	template <int dim>
	typename dealii::FEValuesViews::Vector<dim>::value_type
	FEValues<dim>::vector_value(const typename dealii::FEValuesExtractors::Vector& extractor,
								unsigned int function_no,
								unsigned int point_no){
		return
			(*_fev_space)[extractor].value(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_value(function_no/n_dofs_space_cell,
								   point_no/n_quads_space);
	}

	template <int dim>
	typename dealii::FEValuesViews::Vector<dim>::value_type
	FEValues<dim>::vector_dt(const typename dealii::FEValuesExtractors::Vector& extractor,
							 unsigned int function_no,
							 unsigned int point_no){
		return
			(*_fev_space)[extractor].value(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_grad(function_no/n_dofs_space_cell,
								   point_no/n_quads_space)[0];
	}
	template <int dim>
	typename dealii::FEValuesViews::Vector<dim>::divergence_type
	FEValues<dim>::vector_divergence(const typename dealii::FEValuesExtractors::Vector& extractor,
							 unsigned int function_no,
							 unsigned int point_no){
		return
			(*_fev_space)[extractor].divergence(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_value(function_no/n_dofs_space_cell,
								   point_no/n_quads_space);
	}

	template <int dim>
	typename dealii::FEValuesViews::Vector<dim>::gradient_type
	FEValues<dim>::vector_space_grad(const typename dealii::FEValuesExtractors::Vector& extractor,
									 unsigned int function_no,
									 unsigned int point_no){
		return
			(*_fev_space)[extractor].gradient(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_value(function_no/n_dofs_space_cell,
								   point_no/n_quads_space);
	}

	template <int dim>
	typename dealii::FEValuesViews::Vector<dim>::curl_type
	FEValues<dim>::vector_space_curl(const typename dealii::FEValuesExtractors::Vector& extractor,
									 unsigned int function_no,
									 unsigned int point_no){
		return
			(*_fev_space)[extractor].curl(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_value(function_no/n_dofs_space_cell,
								   point_no/n_quads_space);
	}

	template <int dim>
	double
	FEValues<dim>::time_quadrature_point(unsigned int quadrature_point){
		return
			_fev_time->quadrature_point(quadrature_point/n_quads_space)[0];
	}

	template <int dim>
	dealii::Point<dim>
	FEValues<dim>::space_quadrature_point(unsigned int quadrature_point){
		return
			_fev_space->quadrature_point(quadrature_point%n_quads_space);
	}

	template <int dim>
	double
	FEValues<dim>::JxW(unsigned int quadrature_point){
		return
			_fev_space->JxW(quadrature_point%n_quads_space)*
			_fev_time->JxW(quadrature_point/n_quads_space);
	}

	template <int dim>
	void
	FEValues<dim>::get_local_dof_indices(std::vector<dealii::types::global_dof_index>& indices){
//		Assert(indices.size() == _fe.dofs_per_cell,
//			   dealii::ExcDimensionMismatch(indices.size(),_fe.dofs_per_cell));
		for (unsigned int i = 0 ; i < _fe.dofs_per_cell ; i++){

			indices[i+time_cell_index*_fe.dofs_per_cell] = local_space_dof_index[i%n_dofs_space_cell]
			               + local_time_dof_index[i/n_dofs_space_cell]*n_dofs_space;
		}
	}

	template <int dim>
	std::shared_ptr<dealii::FEValues<dim>>
	FEValues<dim>::spatial(){
		return _fev_space;
	}

	template <int dim>
	std::shared_ptr<dealii::FEValues<1>>
	FEValues<dim>::temporal(){
		return _fev_time;
	}
 ////////////////////////////////////////////////////////////
 // FEJumpValues ////////////////////////////////////////////
 ////////////////////////////////////////////////////////////
	template <int dim>
	FEJumpValues<dim>::FEJumpValues(DG_FiniteElement<dim>& fe,
								    Quadrature<dim>& quad,
							        const dealii::UpdateFlags uflags)
	: _fe(fe),
	  _quad(quad),
	  _fev_space(std::make_shared<dealii::FEValues<dim>>(*fe.spatial(),*quad.spatial(),uflags)),
	  _fev_time(std::make_shared<dealii::FEValues<1>>(*fe.temporal(),dealii::QGaussLobatto<1>(2),uflags)),
	  local_space_dof_index(fe.spatial()->dofs_per_cell),
	  local_time_dof_index(fe.temporal()->dofs_per_cell)
	{n_quadrature_points = _fev_space->n_quadrature_points*_fev_time->n_quadrature_points;}

	template <int dim>
	void
	FEJumpValues<dim>::reinit_space(const typename dealii::TriaIterator<dealii::DoFCellAccessor<dim,dim,false>>& cell_space){
		_fev_space->reinit(cell_space);
		cell_space->get_dof_indices(local_space_dof_index);
	}

	template <int dim>
	void
	FEJumpValues<dim>::reinit_time(const typename dealii::TriaIterator<dealii::DoFCellAccessor<1,1,false>>& cell_time){
		_fev_time->reinit(cell_time);
		cell_time->get_dof_indices(local_time_dof_index);
	}

	template <int dim>
	double
	FEJumpValues<dim>::shape_value_plus(unsigned int function_no,
							   	   	   	unsigned int point_no){
		return
			_fev_space->shape_value(function_no%_fe.spatial()->dofs_per_cell,
									point_no%_fev_space->n_quadrature_points)*
			_fev_time->shape_value(function_no/_fe.spatial()->dofs_per_cell,0);
	}
	template <int dim>
	double
	FEJumpValues<dim>::shape_value_minus(unsigned int function_no,
										unsigned int point_no){
		return
			_fev_space->shape_value(function_no%_fe.spatial()->dofs_per_cell,
									point_no%_fev_space->n_quadrature_points)*
			_fev_time->shape_value(function_no/_fe.spatial()->dofs_per_cell,1);
	}

	template <int dim>
	typename dealii::FEValuesViews::Scalar<dim>::value_type
	FEJumpValues<dim>::scalar_value_plus(const typename dealii::FEValuesExtractors::Scalar& extractor,
										unsigned int function_no,
										unsigned int point_no){
		return
			(*_fev_space)[extractor].value(function_no%_fe.spatial()->dofs_per_cell,
									point_no%_fev_space->n_quadrature_points)*
			_fev_time->shape_value(function_no/_fe.spatial()->dofs_per_cell,0);
	}
	template <int dim>
	typename dealii::FEValuesViews::Scalar<dim>::value_type
	FEJumpValues<dim>::scalar_value_minus(const typename dealii::FEValuesExtractors::Scalar& extractor,
										 unsigned int function_no,
										 unsigned int point_no){
		return
			(*_fev_space)[extractor].value(function_no%_fe.spatial()->dofs_per_cell,
									point_no%_fev_space->n_quadrature_points)*
			_fev_time->shape_value(function_no/_fe.spatial()->dofs_per_cell,1);
	}

	template <int dim>
	typename dealii::FEValuesViews::Vector<dim>::value_type
	FEJumpValues<dim>::vector_value_plus(const typename dealii::FEValuesExtractors::Vector& extractor,
										 unsigned int function_no,
										 unsigned int point_no){
		return
			(*_fev_space)[extractor].value(function_no%_fe.spatial()->dofs_per_cell,
									point_no%_fev_space->n_quadrature_points)*
			_fev_time->shape_value(function_no/_fe.spatial()->dofs_per_cell,0);
	}
	template <int dim>
	typename dealii::FEValuesViews::Vector<dim>::value_type
	FEJumpValues<dim>::vector_value_minus(const typename dealii::FEValuesExtractors::Vector& extractor,
										  unsigned int function_no,
										  unsigned int point_no){
		return
			(*_fev_space)[extractor].value(function_no%_fe.spatial()->dofs_per_cell,
									point_no%_fev_space->n_quadrature_points)*
			_fev_time->shape_value(function_no/_fe.spatial()->dofs_per_cell,1);
	}
	template <int dim>
	double
	FEJumpValues<dim>::JxW(unsigned int quadrature_point){
		return
			_fev_space->JxW(quadrature_point%_fev_space->n_quadrature_points);
	}

	template <int dim>
	std::shared_ptr<dealii::FEValues<dim>>
	FEJumpValues<dim>::spatial(){
		return _fev_space;
	}

	template <int dim>
	std::shared_ptr<dealii::FEValues<1>>
	FEJumpValues<dim>::temporal(){
		return _fev_time;
	}

	////////////////////////////////////////////////////////////
	// FEFaceValues ////////////////////////////////////////////////
	////////////////////////////////////////////////////////////
	template <int dim>
	FEFaceValues<dim>::FEFaceValues(DG_FiniteElement<dim>& fe,
									Quadrature<dim-1>& quad,
									const dealii::UpdateFlags uflags)
	: _fe(fe),
	  _quad(quad),
	  _fev_space(std::make_shared<dealii::FEFaceValues<dim>>(*fe.spatial(),*quad.spatial(),uflags)),
	  _fev_time(std::make_shared<dealii::FEValues<1>>(*fe.temporal(),*quad.temporal(),uflags)),
	  local_space_dof_index(fe.spatial()->dofs_per_cell),
	  local_time_dof_index(fe.temporal()->dofs_per_cell),
	  n_dofs_space(0),
	  time_cell_index(0),
	  n_dofs_space_cell(_fe.spatial()->dofs_per_cell),
	  n_quads_space(_fev_space->n_quadrature_points),
	  n_quadrature_points(_fev_space->n_quadrature_points*_fev_time->n_quadrature_points){}

	template <int dim>
	void
	FEFaceValues<dim>::reinit_space(
			const typename dealii::TriaIterator<dealii::DoFCellAccessor<dim,dim,false>>& cell_space,
			const unsigned int face_no
			){
		_fev_space->reinit(cell_space,face_no);
		cell_space->get_dof_indices(local_space_dof_index);
		n_dofs_space = cell_space->get_dof_handler().n_dofs();
	}

	template <int dim>
	void
	FEFaceValues<dim>::reinit_time(const typename dealii::TriaIterator<dealii::DoFCellAccessor<1,1,false>>& cell_time){
		_fev_time->reinit(cell_time);
		cell_time->get_dof_indices(local_time_dof_index);
		time_cell_index = cell_time->index();
	}

	template <int dim>
	double
	FEFaceValues<dim>::shape_value(unsigned int function_no,
							   unsigned int point_no){
		return
			_fev_space->shape_value(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_value(function_no/n_dofs_space_cell,
								   point_no/n_quads_space);
	}

	template <int dim>
	typename dealii::FEValuesViews::Scalar<dim>::value_type
	FEFaceValues<dim>::scalar_value(const typename dealii::FEValuesExtractors::Scalar& extractor,
								unsigned int function_no,
								unsigned int point_no){
		return
			(*_fev_space)[extractor].value(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_value(function_no/n_dofs_space_cell,
								   point_no/n_quads_space);
	}

	template <int dim>
	typename dealii::FEValuesViews::Vector<dim>::value_type
	FEFaceValues<dim>::vector_value(const typename dealii::FEValuesExtractors::Vector& extractor,
								unsigned int function_no,
								unsigned int point_no){
		return
			(*_fev_space)[extractor].value(function_no%n_dofs_space_cell,
									point_no%n_quads_space)*
			_fev_time->shape_value(function_no/n_dofs_space_cell,
								   point_no/n_quads_space);
	}

	template <int dim>
	double
	FEFaceValues<dim>::time_quadrature_point(unsigned int quadrature_point){
		return
			_fev_time->quadrature_point(quadrature_point/n_quads_space)[0];
	}

	template <int dim>
	dealii::Point<dim>
	FEFaceValues<dim>::space_quadrature_point(unsigned int quadrature_point){
		return
			_fev_space->quadrature_point(quadrature_point%n_quads_space);
	}

	template <int dim>
	const dealii::Tensor<1,dim>&
	FEFaceValues<dim>::space_normal_vector(unsigned int i){
		return
			_fev_space->normal_vector(i%n_quads_space);
	}

	template <int dim>
	double
	FEFaceValues<dim>::JxW(unsigned int quadrature_point){
		return
			_fev_space->JxW(quadrature_point%n_quads_space)*
			_fev_time->JxW(quadrature_point/n_quads_space);
	}

	template <int dim>
	void
	FEFaceValues<dim>::get_local_dof_indices(std::vector<dealii::types::global_dof_index>& indices){
//		Assert(indices.size() == _fe.dofs_per_cell,
//			   dealii::ExcDimensionMismatch(indices.size(),_fe.dofs_per_cell));
		for (unsigned int i = 0 ; i < _fe.dofs_per_cell ; i++){

			indices[i+time_cell_index*_fe.dofs_per_cell] = local_space_dof_index[i%n_dofs_space_cell]
						   + local_time_dof_index[i/n_dofs_space_cell]*n_dofs_space;
		}
	}

	template <int dim>
	std::shared_ptr<dealii::FEFaceValues<dim>>
	FEFaceValues<dim>::spatial(){
		return _fev_space;
	}

	template <int dim>
	std::shared_ptr<dealii::FEValues<1>>
	FEFaceValues<dim>::temporal(){
		return _fev_time;
	}
}}


#include "spacetime_fe_values.inst"

