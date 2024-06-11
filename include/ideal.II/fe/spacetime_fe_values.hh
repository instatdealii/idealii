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

#ifndef INCLUDE_IDEAL_II_FE_SPACETIME_FE_VALUES_HH_
#define INCLUDE_IDEAL_II_FE_SPACETIME_FE_VALUES_HH_

#include <ideal.II/base/quadrature_lib.hh>

#include <ideal.II/fe/fe_dg.hh>

#include <deal.II/fe/fe_values.h>

namespace idealii::spacetime
{

  /**
   * @brief Evaluation of the tensor-product space-time basis functions.
   *
   * This class supplies common derivatives of the space-time basis functions by
   * multiplying the corresponding spatial and temporal basis functions in the
   * given quadrature points.
   *
   * In practice spatial derivatives are handled by the underlying spatial
   * dealii::FEValues object and temporal derivatives are handled by the
   * underlying temporal dealii::FEValues.
   */
  template <int dim>
  class FEValues
  {
  public:
    /**
     * @brief Constructor of the FEValues class.
     *
     * @param fe: The underlying space-time finite element description class.
     * @param quad: The space-time quadrature formula to be used.
     * @param uflags: The update flags to be used during the reinit calls.
     */
    FEValues(DG_FiniteElement<dim>    &fe,
             Quadrature<dim>          &quad,
             const dealii::UpdateFlags uflags);

    /**
     * @brief Reinitialize all objects of the underlying spatial FEValues object.
     * This function calls reinit(cell_space) of the spatial FEValues object.
     * @param cell_space Iterator pointing to the current element in space.
     */
    void
    reinit_space(const typename dealii::TriaIterator<
                 dealii::DoFCellAccessor<dim, dim, false>> &cell_space);
    /**
     * @brief Reinitialize all objects of the underlying temporal FEValues object.
     * This function calls reinit(cell_time) of the temporal FEValues object.
     * @param cell_time Iterator pointing to the current element in time.
     */
    void
    reinit_time(
      const typename dealii::TriaIterator<dealii::DoFCellAccessor<1, 1, false>>
        &cell_time);

    /**
     * @brief Value of the space-time shape function at spacetime-quadrature point.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    double
    shape_value(unsigned int function_no, unsigned int point_no);

    /**
     * @brief Temporal derivative of the space-time shape function at spacetime-quadrature point.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    double
    shape_dt(unsigned int function_no, unsigned int point_no);

    /**
     * @brief Spatial derivative of the space-time shape function at spacetime-quadrature point.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    dealii::Tensor<1, dim>
    shape_space_grad(unsigned int function_no, unsigned int point_no);

    /**
     * @brief Function values of a given vector at all quadrature points
     * @in fe_function
     * @out values
     */
    template <class InputVector>
    void
    get_function_values(
      const InputVector &fe_function,
      std::vector<dealii::Vector<typename InputVector::value_type>> &values)
      const;

    /**
     * @brief Function values of a given vector at all quadrature points
     * @in fe_function
     * @out values
     */
    template <class InputVector>
    void
    get_function_dt(
      const InputVector &fe_function,
      std::vector<dealii::Vector<typename InputVector::value_type>> &values)
      const;

    /**
     * @brief Spatial function gradients of a given vector at all quadrature points
     * @in fe_function
     * @out values
     */
    template <class InputVector>
    void
    get_function_space_gradients(
      const InputVector &fe_function,
      std::vector<
        std::vector<dealii::Tensor<1, dim, typename InputVector::value_type>>>
        &gradients) const;


    /**
     * @brief Value of the space-time shape function of a scalar finite element component.
     *
     * This function passes the extractor to the underlying spatial FEValues
     * object and then calls the value function of the resulting view.
     * @param extractor Scalar extractor defining the scalar finite element component to be evaluated.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Scalar<dim>::value_type
    scalar_value(const typename dealii::FEValuesExtractors::Scalar &extractor,
                 unsigned int                                       function_no,
                 unsigned int                                       point_no);

    /**
     * @brief Temporal derivative of the space-time shape function of a scalar finite element component.
     *
     * This function passes the extractor to the underlying spatial FEValues
     * object and then calls the value function of the resulting view.
     * @param extractor Scalar extractor defining the scalar finite element component to be evaluated.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Scalar<dim>::value_type
    scalar_dt(const typename dealii::FEValuesExtractors::Scalar &extractor,
              unsigned int                                       function_no,
              unsigned int                                       point_no);
    /**
     * @brief Spatial derivative of the space-time shape function of a scalar finite element component.
     *
     * This function passes the extractor to the underlying spatial FEValues
     * object and then calls the gradient function of the resulting view.
     * @param extractor Scalar extractor defining the scalar finite element component to be evaluated.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Scalar<dim>::gradient_type
    scalar_space_grad(
      const typename dealii::FEValuesExtractors::Scalar &extractor,
      unsigned int                                       function_no,
      unsigned int                                       point_no);

    /**
     * @brief Value of the space-time shape function of a vector-valued finite element component.
     *
     * This function passes the extractor to the underlying spatial FEValues
     * object and then calls the value function of the resulting view.
     * @param extractor Scalar extractor defining the scalar finite element component to be evaluated.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Vector<dim>::value_type
    vector_value(const typename dealii::FEValuesExtractors::Vector &extractor,
                 unsigned int                                       function_no,
                 unsigned int                                       point_no);

    /**
     * @brief Temporal derivative of the space-time shape function of a vector-valued finite element component.
     *
     * This function passes the extractor to the underlying spatial FEValues
     * object and then calls the value function of the resulting view.
     * @param extractor Scalar extractor defining the scalar finite element component to be evaluated.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Vector<dim>::value_type
    vector_dt(const typename dealii::FEValuesExtractors::Vector &extractor,
              unsigned int                                       function_no,
              unsigned int                                       point_no);

    /**
     * @brief Spatial divergence of the space-time shape function of a vector-valued finite element component.
     *
     * This function passes the extractor to the underlying spatial FEValues
     * object and then calls the value function of the resulting view.
     * @param extractor Scalar extractor defining the scalar finite element component to be evaluated.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Vector<dim>::divergence_type
    vector_divergence(
      const typename dealii::FEValuesExtractors::Vector &extractor,
      unsigned int                                       function_no,
      unsigned int                                       point_no);

    /**
     * @brief Spatial gradient of the space-time shape function of a vector-valued finite element component.
     *
     * This function passes the extractor to the underlying spatial FEValues
     * object and then calls the gradient function of the resulting view.
     * @param extractor Vector extractor defining the vector finite element component to be evaluated.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Vector<dim>::gradient_type
    vector_space_grad(
      const typename dealii::FEValuesExtractors::Vector &extractor,
      unsigned int                                       function_no,
      unsigned int                                       point_no);

    /**
     * @brief Spatial curl of the space-time shape function of a vector-valued finite element component.
     *
     * This function passes the extractor to the underlying spatial FEValues
     * object and then calls the curl function of the resulting view.
     * @param extractor Vector extractor defining the vector finite element component to be evaluated.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Vector<dim>::curl_type
    vector_space_curl(
      const typename dealii::FEValuesExtractors::Vector &extractor,
      unsigned int                                       function_no,
      unsigned int                                       point_no);

    /**
     * @brief Get the temporal quadrature point of the given space-time quadrature index.
     * @param quadrature_point space-time quadrature index.
     */
    double
    time_quadrature_point(unsigned int quadrature_point);

    /**
     * @brief Get the spatial quadrature point of the given space-time quadrature index.
     * @param quadrature_point space-time quadrature index.
     */
    dealii::Point<dim>
    space_quadrature_point(unsigned int quadrature_point);

    /**
     * @brief Mapped space-time quadrature weight.
     * @param quadrature_point space-time quadrature index.
     */
    double
    JxW(const unsigned int quadrature_point);
    /**
     * @brief Local space-time DoF indices of the current space-time element.
     * @param A vector of indices to save the result to.
     */
    void
    get_local_dof_indices(
      std::vector<dealii::types::global_dof_index> &indices);

    /**
     * @brief The underlying spatial FEValues object.
     * @return A shared pointer to the spatial FEValues object.
     */
    std::shared_ptr<dealii::FEValues<dim>>
    spatial();
    /**
     * @brief The underlying temporal FEValues object.
     * @return A shared pointer to the temporal FEValues object.
     */
    std::shared_ptr<dealii::FEValues<1>>
    temporal();

  private:
    DG_FiniteElement<dim> &_fe;
    Quadrature<dim>       &_quad;

    std::shared_ptr<dealii::FEValues<dim>> _fev_space;
    std::shared_ptr<dealii::FEValues<1>>   _fev_time;

    std::vector<dealii::types::global_dof_index> local_space_dof_index;
    std::vector<dealii::types::global_dof_index> local_time_dof_index;

    unsigned int       n_dofs_space;
    unsigned int       time_cell_index;
    const unsigned int n_dofs_space_cell;
    const unsigned int n_quads_space;

  public:
    /**
     * @brief Number of space-time quadrature points per element.
     */
    unsigned int n_quadrature_points;
  };

  /**
   * @brief Evaluation of the tensor-product space-time basis functions at the temporal element edges.
   *
   * This class supplies the limits from above and below of the space-time basis
   * functions by multiplying the corresponding spatial basis functions in the
   * given spatial quadrature points by the left or right face_values of the
   * temporal basis functions.
   *
   * In practice spatial values are handled by the underlying spatial
   * dealii::FEValues object and temporal limits are handled by the underlying
   * temporal dealii::FEValues with two point Lobatto quadrature.
   */
  template <int dim>
  class FEJumpValues
  {
  public:
    /**
     * @brief Constructor of the FEJumpValues class.
     *
     * @param fe: The underlying space-time finite element description class.
     * @param quad: The space-time quadrature formula to be used.
     * @param uflags: The update flags to be used during the reinit calls.
     */
    FEJumpValues(DG_FiniteElement<dim>    &fe,
                 Quadrature<dim>          &quad,
                 const dealii::UpdateFlags uflags);

    /**
     * @brief Reinitialize all objects of the underlying spatial FEValues object.
     * This function calls reinit(cell_space) of the spatial FEValues object.
     * @param cell_space Iterator pointing to the current element in space.
     */
    void
    reinit_space(const typename dealii::TriaIterator<
                 dealii::DoFCellAccessor<dim, dim, false>> &cell_space);

    /**
     * @brief Reinitialize all objects of the underlying temporal FEValues object.
     * This function calls reinit(cell_time) of the temporal FEValues object.
     * @param cell_time Iterator pointing to the current element in time.
     */
    void
    reinit_time(
      const typename dealii::TriaIterator<dealii::DoFCellAccessor<1, 1, false>>
        &cell_time);

    /**
     * @brief Value of the limit from above of the space-time shape function at the spatial quadrature point.
     *
     * The temporal value is evaluated at the left point of the unit element
     * i.e. 0.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    double
    shape_value_plus(unsigned int function_no, unsigned int point_no);

    /**
     * @brief Value of the limit from below of the space-time shape function at the spatial quadrature point.
     *
     * The temporal value is evaluated at the right point of the unit element
     * i.e. 1.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    double
    shape_value_minus(unsigned int function_no, unsigned int point_no);

    /**
     * @brief Left temporal limit from below of function values of a given vector at all space quadrature points
     * @in fe_function
     * @out values
     */
    template <class InputVector>
    void
    get_function_values_minus(
      const InputVector &fe_function,
      std::vector<dealii::Vector<typename InputVector::value_type>> &values)
      const;

    /**
     * @brief Left temporal limit from above of function values of a given vector at all space quadrature points
     * @in fe_function
     * @out values
     */
    template <class InputVector>
    void
    get_function_values_plus(
      const InputVector &fe_function,
      std::vector<dealii::Vector<typename InputVector::value_type>> &values)
      const;

    /**
     * @brief Value of the limit from above of the space-time shape function of a scalar finite element component.
     *
     * The temporal value is evaluated at the left point of the unit element
     * i.e. 0.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Scalar<dim>::value_type
    scalar_value_plus(
      const typename dealii::FEValuesExtractors::Scalar &extractor,
      unsigned int                                       function_no,
      unsigned int                                       point_no);

    /**
     * @brief Value of the limit from bewlo of the space-time shape function of a scalar finite element component.
     *
     * The temporal value is evaluated at the right point of the unit element
     * i.e. 1.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Scalar<dim>::value_type
    scalar_value_minus(
      const typename dealii::FEValuesExtractors::Scalar &extractor,
      unsigned int                                       function_no,
      unsigned int                                       point_no);

    /**
     * @brief Value of the limit from above of the space-time shape function of a vector-valued finite element component.
     *
     * The temporal value is evaluated at the left point of the unit element
     * i.e. 0.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Vector<dim>::value_type
    vector_value_plus(
      const typename dealii::FEValuesExtractors::Vector &extractor,
      unsigned int                                       function_no,
      unsigned int                                       points_no);

    /**
     * @brief Value of the limit from below of the space-time shape function of a vector-valued finite element component.
     *
     * The temporal value is evaluated at the right point of the unit element
     * i.e. 1.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Vector<dim>::value_type
    vector_value_minus(
      const typename dealii::FEValuesExtractors::Vector &extractor,
      unsigned int                                       function_no,
      unsigned int                                       point_no);
    /**
     * @brief The underlying spatial FEValues object.
     * @return A shared pointer to the spatial FEValues object.
     */
    std::shared_ptr<dealii::FEValues<dim>>
    spatial();

    /**
     * @brief The underlying temporal FEValues object.
     * @return A shared pointer to the temporal FEValues object.
     */
    std::shared_ptr<dealii::FEValues<1>>
    temporal();

    /**
     * @brief Mapped space-time quadrature weight.
     * @param quadrature_point space-time quadrature index.
     */
    double
    JxW(const unsigned int quadrature_point);

    /**
     * @brief Number of spatial quadrature points per element.
     */
    unsigned int n_quadrature_points;

  private:
    unsigned int           n_dofs_space;
    DG_FiniteElement<dim> &_fe;
    Quadrature<dim>       &_quad;

    std::shared_ptr<dealii::FEValues<dim>> _fev_space;
    std::shared_ptr<dealii::FEValues<1>>   _fev_time;

    std::vector<dealii::types::global_dof_index> local_space_dof_index;
    std::vector<dealii::types::global_dof_index> local_time_dof_index;
  };

  /**
   * @brief Evaluation of the tensor-product space-time basis functions on spatial element faces.
   *
   * This class supplies common derivatives of the space-time basis functions by
   * multiplying the corresponding spatial and temporal basis functions in the
   * given quadrature points.
   *
   * In practice spatial derivatives are handled by the underlying spatial
   * dealii::FEFaceValues object and temporal derivatives are handled by the
   * underlying temporal dealii::FEValues.
   */
  template <int dim>
  class FEFaceValues
  {
  public:
    /**
     * @brief Constructor of the FEValues class.
     *
     * @param fe: The underlying space-time finite element description class.
     * @param quad: The space-time quadrature formula to be used.
     * @param uflags: The update flags to be used during the reinit calls.
     * @param additional_flags: Additional update flags that are only appended
     *                          to the FEFace Object (e.g.:
     * update_normal_vectors)
     */
    FEFaceValues(DG_FiniteElement<dim>    &fe,
                 Quadrature<dim - 1>      &quad,
                 const dealii::UpdateFlags uflags,
                 const dealii::UpdateFlags additional_flags);

    /**
     * @brief Reinitialize all objects of the underlying spatial FEValues object.
     * This function calls reinit(cell_space) of the spatial FEValues object.
     * @param cell_space Iterator pointing to the current element in space.
     */
    void
    reinit_space(const typename dealii::TriaIterator<
                   dealii::DoFCellAccessor<dim, dim, false>> &cell_space,
                 const unsigned int                           face_no);
    /**
     * @brief Reinitialize all objects of the underlying temporal FEValues object.
     * This function calls reinit(cell_time) of the temporal FEValues object.
     * @param cell_time Iterator pointing to the current element in time.
     */
    void
    reinit_time(
      const typename dealii::TriaIterator<dealii::DoFCellAccessor<1, 1, false>>
        &cell_time);

    /**
     * @brief Value of the space-time shape function at spacetime-quadrature point.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    double
    shape_value(unsigned int function_no, unsigned int point_no);

    /**
     * @brief Value of the space-time shape function of a scalar finite element component.
     *
     * This function passes the extractor to the underlying spatial FEValues
     * object and then calls the value function of the resulting view.
     * @param extractor Scalar extractor defining the scalar finite element component to be evaluated.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Scalar<dim>::value_type
    scalar_value(const typename dealii::FEValuesExtractors::Scalar &extractor,
                 unsigned int                                       function_no,
                 unsigned int                                       point_no);

    /**
     * @brief Value of the space-time shape function of a vector-valued finite element component.
     *
     * This function passes the extractor to the underlying spatial FEValues
     * object and then calls the value function of the resulting view.
     * @param extractor Scalar extractor defining the scalar finite element component to be evaluated.
     * @param function_no The number of the space-time function/dof to be evaluated.
     * @param point_no The number of the quadrature point to evaluate at.
     */
    typename dealii::FEValuesViews::Vector<dim>::value_type
    vector_value(const typename dealii::FEValuesExtractors::Vector &extractor,
                 unsigned int                                       function_no,
                 unsigned int                                       point_no);

    /**
     * @brief Get the temporal quadrature point of the given space-time quadrature index.
     * @param quadrature_point space-time quadrature index.
     */
    double
    time_quadrature_point(unsigned int quadrature_point);

    /**
     * @brief Get the spatial quadrature point of the given space-time quadrature index.
     * @param quadrature_point space-time quadrature index.
     */
    dealii::Point<dim>
    space_quadrature_point(unsigned int quadrature_point);

    /**
     * @brief Get the normal vector at the spatial face.
     * @param i The space-time index of the quadrature point
     */
    const dealii::Tensor<1, dim> &
    space_normal_vector(unsigned int i);

    /**
     * @brief Mapped space-time quadrature weight.
     * @param quadrature_point space-time quadrature index.
     */
    double
    JxW(const unsigned int quadrature_point);
    /**
     * @brief Local space-time DoF indices of the current space-time element.
     * @param A vector of indices to save the result to.
     */
    void
    get_local_dof_indices(
      std::vector<dealii::types::global_dof_index> &indices);

    /**
     * @brief The underlying spatial FEValues object.
     * @return A shared pointer to the spatial FEValues object.
     */
    std::shared_ptr<dealii::FEFaceValues<dim>>
    spatial();
    /**
     * @brief The underlying temporal FEValues object.
     * @return A shared pointer to the temporal FEValues object.
     */
    std::shared_ptr<dealii::FEValues<1>>
    temporal();

  private:
    DG_FiniteElement<dim> &_fe;
    Quadrature<dim - 1>   &_quad;

    std::shared_ptr<dealii::FEFaceValues<dim>> _fev_space;
    std::shared_ptr<dealii::FEValues<1>>       _fev_time;

    std::vector<dealii::types::global_dof_index> local_space_dof_index;
    std::vector<dealii::types::global_dof_index> local_time_dof_index;

    unsigned int       n_dofs_space;
    unsigned int       time_cell_index;
    const unsigned int n_dofs_space_cell;
    const unsigned int n_quads_space;

  public:
    /**
     * @brief Number of space-time quadrature points per element.
     */
    unsigned int n_quadrature_points;
  };
} // namespace idealii::spacetime

#endif /* INCLUDE_IDEAL_II_BASE_SPACETIME_QUADRATURE_HH_ */
