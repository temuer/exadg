/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <exadg/structure/material/library/surfactant.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>
#include <cmath>

#include <exadg/structure/material/library/alveolar_tissue.h>

namespace ExaDG
{
namespace Structure
{

template<int dim, typename Number>
WiechertAlveolarTissue<dim, Number>::WiechertAlveolarTissue(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      dof_index,
  unsigned int const                      quad_index,
  WiechertAlveolarTissueData<dim> const & data)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    surfactant_model(data.surfactant_data, matrix_free.n_boundary_face_batches())
{
}

template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  return second_piola_kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_eval(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  (void)cell;
  (void)q;

  tensor const F = dealii::Physics::Elasticity::Kinematics::F(gradient_displacement);
  scalar const J = dealii::determinant(F);

  symmetric_tensor const C     = dealii::Physics::Elasticity::Kinematics::C(F);
  symmetric_tensor const C_inv = dealii::invert(C);
  scalar const           I_1   = dealii::trace(C);

  scalar const Jpow_minus_two_thirds = std::pow(J, static_cast<Number>(-TWO_THIRDS));

  // Ground substance
  symmetric_tensor S = (data.shear_modulus * Jpow_minus_two_thirds) *
                       (get_identity_symmetric_tensor<dim, Number>() - ONE_THIRD * I_1 * C_inv);

  // Incompressibility penalty
  scalar const Jpow_two_exponent =
    std::pow(J, static_cast<Number>(2.0 * data.incompressibility_exponent));

  S += 2.0 * data.incompressibility_penalty * data.incompressibility_exponent *
       (Jpow_two_exponent - 1.0 / Jpow_two_exponent) * C_inv;

  // Fibers (serialize due to the conditional)
  scalar const fiber_mask = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
    I_1,
    dealii::make_vectorized_array<Number>(3.0),
    dealii::make_vectorized_array<Number>(0.0),
    dealii::make_vectorized_array<Number>(1.0));

  scalar const vol_strain   = ONE_THIRD * I_1 - 1.0;
  scalar const fiber_stress = fiber_mask * TWO_THIRDS * data.fiber_k_1 * vol_strain *
                              std::exp(data.fiber_k_2 * vol_strain * vol_strain);

  // Update diagonal entries
  for(int d{0}; d < dim; d++)
  {
    S[d][d] += fiber_stress;
  }

  return S;
}

template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(unsigned int const cell,
                                                                   unsigned int const q) const
  -> symmetric_tensor
{
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                 "this material does not store tensorial quantities."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
  tensor const &     gradient_increment,
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  (void)cell;
  (void)q;

  tensor const F     = dealii::Physics::Elasticity::Kinematics::F(gradient_displacement);
  scalar const J     = dealii::determinant(F);
  tensor const F_inv = dealii::invert(F);

  symmetric_tensor const C     = dealii::Physics::Elasticity::Kinematics::C(F);
  symmetric_tensor const C_inv = compute_C_inv(F_inv);
  scalar const           I_1   = dealii::trace(C);

  scalar const           Du_I_1   = 2.0 * dealii::trace(dealii::transpose(F) * gradient_increment);
  tensor const           Du_F_inv = -F_inv * gradient_increment * F_inv;
  symmetric_tensor const Du_C_inv = compute_H_plus_HT(Du_F_inv * dealii::transpose(F_inv));
  scalar const           Du_J_over_J = dealii::trace(gradient_increment * F_inv);

  scalar const Jpow_minus_two_thirds = std::pow(J, static_cast<Number>(-TWO_THIRDS));

  symmetric_tensor Du_S;

  // Ground substance
  {
    scalar const tmp = ONE_THIRD * data.shear_modulus * Jpow_minus_two_thirds;

    // J term
    Du_S += -2.0 * tmp * Du_J_over_J *
            (get_identity_symmetric_tensor<dim, Number>() - ONE_THIRD * I_1 * C_inv);

    // I_1 term
    Du_S += -tmp * Du_I_1 * C_inv;

    // C_inv term
    Du_S += -tmp * I_1 * Du_C_inv;
  }

  // Incompressibility penalty
  {
    scalar const tmp = std::pow(J, static_cast<Number>(2.0 * data.incompressibility_exponent));

    // J term
    Du_S += 4.0 * data.incompressibility_penalty * data.incompressibility_exponent *
            data.incompressibility_exponent * (tmp + 1.0 / tmp) * Du_J_over_J * C_inv;

    // C_inv term
    Du_S += 2.0 * data.incompressibility_penalty * data.incompressibility_exponent *
            (tmp - 1.0 / tmp) * Du_C_inv;
  }


  scalar const fiber_mask = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
    I_1,
    dealii::make_vectorized_array<Number>(3.0),
    dealii::make_vectorized_array<Number>(0.0),
    dealii::make_vectorized_array<Number>(1.0));

  scalar const vol_strain = ONE_THIRD * I_1 - 1.0;
  scalar const tmp        = data.fiber_k_2 * vol_strain * vol_strain;

  scalar const fiber_stress_increment =
    fiber_mask * TWO_NINTHS * data.fiber_k_1 * std::exp(tmp) * (1.0 + 2.0 * tmp) * Du_I_1;

  for(int d{0}; d < dim; d++)
  {
    Du_S[d][d] += fiber_stress_increment;
  }

  return Du_S;
}



template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::kirchhoff_stress(tensor const &     gradient_displacement,
                                                      unsigned int const cell,
                                                      unsigned int const q) const
  -> symmetric_tensor
{
  return kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::kirchhoff_stress_eval(tensor const &     gradient_displacement,
                                                           unsigned int const cell,
                                                           unsigned int const q) const
  -> symmetric_tensor
{
  (void)gradient_displacement;
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This material does not (yet) support spatial integration."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::kirchhoff_stress(unsigned int const cell,
                                                      unsigned int const q) const
  -> symmetric_tensor
{
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                 "this material does not store tensorial quantities."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::contract_with_J_times_C(
  symmetric_tensor const & symmetric_gradient_increment,
  tensor const &           gradient_displacement,
  unsigned int const       cell,
  unsigned int const       q) const -> symmetric_tensor
{
  (void)symmetric_gradient_increment;
  (void)gradient_displacement;
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This material does not (yet) support spatial integration."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}


template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::contract_with_J_times_C(
  symmetric_tensor const & symmetric_gradient_increment,
  unsigned int const       cell,
  unsigned int const       q) const -> symmetric_tensor
{
  (void)symmetric_gradient_increment;
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This function cannot be called with `non-caching` material."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}


template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::do_set_cell_linearization_data(
  std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
  unsigned int const                                                         cell) const -> void
{
  (void)integrator_lin;
  (void)cell;

  AssertThrow(false, dealii::ExcMessage("This material does not support caching."));

  return;
}

template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::one_over_J(unsigned int const cell, unsigned int const q) const
  -> scalar
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed one_over_J."));

  return dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN());
}

template<int dim, typename Number>
auto
WiechertAlveolarTissue<dim, Number>::gradient_displacement(unsigned int const cell,
                                                           unsigned int const q) const -> tensor
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed deformation gradient."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_tensor<dim, Number>());
}


template<int dim, typename Number>
RauschAlveolarTissue<dim, Number>::RauschAlveolarTissue(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      dof_index,
  unsigned int const                      quad_index,
  RauschAlveolarTissueData<dim> const &   data)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    surfactant_model(data.surfactant_data, matrix_free.n_boundary_face_batches())
{
}

template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  return second_piola_kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_eval(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  (void)cell;
  (void)q;

  tensor const           F     = dealii::Physics::Elasticity::Kinematics::F(gradient_displacement);
  scalar const           J     = dealii::determinant(F);
  symmetric_tensor const C     = dealii::Physics::Elasticity::Kinematics::C(F);
  symmetric_tensor const C_inv = dealii::invert(C);

  return 0.5 * data.E / (1 + data.nu) *
         (get_identity_symmetric_tensor<dim, Number>() -
          std::pow(J, static_cast<Number>(2.0 * -data.nu / (1.0 - 2.0 * data.nu))) * C_inv);
}

template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(unsigned int const cell,
                                                                 unsigned int const q) const
  -> symmetric_tensor
{
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                 "this material does not store tensorial quantities."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
  tensor const &     gradient_increment,
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  (void)cell;
  (void)q;

  tensor const F     = dealii::Physics::Elasticity::Kinematics::F(gradient_displacement);
  scalar const J     = dealii::determinant(F);
  tensor const F_inv = dealii::invert(F);

  symmetric_tensor const C_inv = compute_C_inv(F_inv);

  tensor const           Du_F_inv    = -F_inv * gradient_increment * F_inv;
  symmetric_tensor const Du_C_inv    = compute_H_plus_HT(Du_F_inv * dealii::transpose(F_inv));
  scalar const           Du_J_over_J = dealii::trace(gradient_increment * F_inv);


  symmetric_tensor Du_S;

  // J term
  Du_S += -data.E * data.nu / (1.0 + data.nu) / (1.0 - 2.0 * data.nu) *
          std::pow(J, static_cast<Number>(-2.0 * data.nu / (1.0 - 2.0 * data.nu))) * Du_J_over_J *
          C_inv;

  // C_inv term
  Du_S += 0.5 * data.E / (1.0 + data.nu) *
          std::pow(J, static_cast<Number>(-2.0 * data.nu / (1.0 - 2.0 * data.nu))) * Du_C_inv;


  return Du_S;
}



template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::kirchhoff_stress(tensor const &     gradient_displacement,
                                                    unsigned int const cell,
                                                    unsigned int const q) const -> symmetric_tensor
{
  return kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::kirchhoff_stress_eval(tensor const &     gradient_displacement,
                                                         unsigned int const cell,
                                                         unsigned int const q) const
  -> symmetric_tensor
{
  (void)gradient_displacement;
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This material does not (yet) support spatial integration."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::kirchhoff_stress(unsigned int const cell,
                                                    unsigned int const q) const -> symmetric_tensor
{
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                 "this material does not store tensorial quantities."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::contract_with_J_times_C(
  symmetric_tensor const & symmetric_gradient_increment,
  tensor const &           gradient_displacement,
  unsigned int const       cell,
  unsigned int const       q) const -> symmetric_tensor
{
  (void)symmetric_gradient_increment;
  (void)gradient_displacement;
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This material does not (yet) support spatial integration."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}


template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::contract_with_J_times_C(
  symmetric_tensor const & symmetric_gradient_increment,
  unsigned int const       cell,
  unsigned int const       q) const -> symmetric_tensor
{
  (void)symmetric_gradient_increment;
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This function cannot be called with `non-caching` material."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}


template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::do_set_cell_linearization_data(
  std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
  unsigned int const                                                         cell) const -> void
{
  (void)integrator_lin;
  (void)cell;

  AssertThrow(false, dealii::ExcMessage("This material does not support caching."));

  return;
}

template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::one_over_J(unsigned int const cell, unsigned int const q) const
  -> scalar
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed one_over_J."));

  return dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN());
}

template<int dim, typename Number>
auto
RauschAlveolarTissue<dim, Number>::gradient_displacement(unsigned int const cell,
                                                         unsigned int const q) const -> tensor
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed deformation gradient."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_tensor<dim, Number>());
}

template class WiechertAlveolarTissue<2, float>;
template class WiechertAlveolarTissue<3, float>;

template class WiechertAlveolarTissue<2, double>;
template class WiechertAlveolarTissue<3, double>;

template class RauschAlveolarTissue<2, float>;
template class RauschAlveolarTissue<3, float>;

template class RauschAlveolarTissue<2, double>;
template class RauschAlveolarTissue<3, double>;

} // namespace Structure
} // namespace ExaDG