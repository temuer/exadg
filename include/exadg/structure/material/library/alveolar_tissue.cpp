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
#include <exadg/structure/material/library/alveolar_tissue.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>
#include <cmath>
#include <cstddef>
#include <limits>

namespace ExaDG
{
namespace Structure
{

template<int dim, typename Number>
AlveolarTissue<dim, Number>::AlveolarTissue(dealii::MatrixFree<dim, Number> const & matrix_free,
                                            unsigned int const                      dof_index,
                                            unsigned int const                      quad_index,
                                            AlveolarTissueData<dim> const &         data)
  : dof_index(dof_index), quad_index(quad_index), data(data)
{
  surfactant_model_coefficients.reinit(matrix_free.n_boundary_face_batches(),
                                       matrix_free.get_n_q_points_face(quad_index));

  surfactant_model_coefficients.fill(std::array<scalar, 2>{{
    dealii::make_vectorized_array<Number>(0.0),
    dealii::make_vectorized_array<Number>(0.0),
  }});
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::surface_tension(scalar const &     da_dA,
                                             double const       time_step_size,
                                             unsigned int const boundary_face,
                                             unsigned int const q) const -> scalar
{
  std::array<scalar, 2> const & coeffs{surfactant_model_coefficients[boundary_face][q]};

  scalar gamma = dealii::make_vectorized_array<Number>(0.0);

  for(std::size_t v{0}; v < scalar::size(); v++)
  {
    Number const relative_concentration_old = coeffs[0][v];
    Number const da_dA_old                  = coeffs[1][v];

    if(relative_concentration_old < 1.0)
    // Regime 1
    {
      Number const inv_delta_t = 1.0f / static_cast<Number>(time_step_size);
      Number const k1_C =
        static_cast<Number>(data.surfactant_k_1) * static_cast<Number>(data.surfactant_c);
      Number const k2               = static_cast<Number>(data.surfactant_k_2);
      Number const concentration_eq = static_cast<Number>(data.surfactant_concentration_eq);

      Number const relative_concentration_new =
        (relative_concentration_old * da_dA_old * inv_delta_t +
         da_dA[v] * concentration_eq * k1_C) /
        da_dA[v] * (inv_delta_t + k1_C + k2);

      gamma[v] = data.surface_tension_ref - data.surfactant_m_1 * relative_concentration_new /
                                              data.surfactant_concentration_eq;
    }
    else if(relative_concentration_old <
            (data.surfactant_concentration_max / data.surfactant_concentration_eq))
    // Regime 2
    {
      Number const relative_concentration_new = relative_concentration_old * da_dA_old / da_dA[v];
      gamma[v] =
        data.surface_tension_eq -
        data.surfactant_m_2 * (relative_concentration_new / data.surfactant_concentration_eq - 1.0);
    }
    else
    // Regime 3
    {
      gamma[v] = data.surface_tension_min;
    }
  }

  return gamma;
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(tensor const &     gradient_displacement,
                                                           unsigned int const cell,
                                                           unsigned int const q) const
  -> symmetric_tensor
{
  return second_piola_kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_eval(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
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
  for(std::size_t v{0}; v < scalar::size(); v++)
  {
    Number const I_1_v{I_1[v]};

    // Conditional
    if(I_1_v < 3.0)
    {
      continue;
    }

    Number const vol_strain = static_cast<Number>(ONE_THIRD * I_1_v - 1.0);

    // Update diagonal entries
    for(int d{0}; d < dim; d++)
    {
      S[d][d][v] += ONE_THIRD * 2.0 * data.fiber_k_1 * vol_strain *
                    std::exp(data.fiber_k_2 * vol_strain * vol_strain);
    }
  }

  return S;
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(unsigned int const cell,
                                                           unsigned int const q) const
  -> symmetric_tensor
{
  AssertThrow(false,
              dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                 "this material does not store tensorial quantities."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
  tensor const &     gradient_increment,
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  tensor const F     = dealii::Physics::Elasticity::Kinematics::F(gradient_displacement);
  scalar const J     = dealii::determinant(F);
  tensor const F_inv = dealii::invert(F);

  symmetric_tensor const C     = dealii::Physics::Elasticity::Kinematics::C(F);
  symmetric_tensor const C_inv = compute_C_inv(F_inv);
  scalar const           I_1   = dealii::trace(C);

  scalar const           Du_I_1   = 2.0 * dealii::trace(dealii::transpose(F) * gradient_increment);
  tensor const           Du_F_inv = -F_inv * gradient_increment * F_inv;
  symmetric_tensor const Du_C_inv = compute_H_plus_HT(Du_F_inv * dealii::transpose(F_inv));

  // TODO! ask about this.
  scalar const Du_J_over_J = dealii::trace(gradient_increment * F_inv);

  scalar const Jpow_minus_two_thirds = std::pow(J, static_cast<Number>(-TWO_THIRDS));

  // Ground substance
  symmetric_tensor Du_S{-TWO_THIRDS * data.shear_modulus * Jpow_minus_two_thirds * Du_J_over_J *
                          get_identity_symmetric_tensor<dim, Number>() -
                        ONE_THIRD * data.shear_modulus * Jpow_minus_two_thirds * I_1 * Du_C_inv +
                        Du_J_over_J * TWO_NINTHS * data.shear_modulus * Jpow_minus_two_thirds *
                          I_1 * C_inv -
                        ONE_THIRD * data.shear_modulus * Jpow_minus_two_thirds * Du_I_1 * C_inv};

  // Incompressibility penalty
  scalar const Jpow_two_exponent =
    std::pow(J, static_cast<Number>(2.0 * data.incompressibility_exponent));

  Du_S += 2.0 * data.incompressibility_penalty * data.incompressibility_exponent *
          (Jpow_two_exponent - 1.0 / Jpow_two_exponent) *
          (Du_C_inv + 2.0 * data.incompressibility_exponent * Du_J_over_J * C_inv);

  // TODO! -> SIMDComparison
  // Fibers (serialize due to the conditional)
  for(std::size_t v{0}; v < scalar::size(); v++)
  {
    Number const I_1_v{I_1[v]};

    // Conditional
    if(I_1_v < 3.0)
    {
      continue;
    }

    Number const vol_strain{static_cast<Number>(ONE_THIRD * I_1_v - 1.0)};
    Number const k2_times_vol_strain_pow2{static_cast<Number>(data.fiber_k_2) * vol_strain *
                                          vol_strain};

    // Update diagonal entries
    for(int d{0}; d < dim; d++)
    {
      Du_S[d][d][v] += TWO_NINTHS * data.fiber_k_1 * std::exp(k2_times_vol_strain_pow2) *
                       (1.0 + 2.0 * k2_times_vol_strain_pow2);
    }
  }

  return Du_S;
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::kirchhoff_stress(tensor const &     gradient_displacement,
                                              unsigned int const cell,
                                              unsigned int const q) const -> symmetric_tensor
{
  return kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::kirchhoff_stress_eval(tensor const &     gradient_displacement,
                                                   unsigned int const cell,
                                                   unsigned int const q) const -> symmetric_tensor
{
  // tensor const           F{dealii::Physics::Elasticity::Kinematics::F(gradient_displacement)};
  // symmetric_tensor const S{second_piola_kirchhoff_stress_eval(gradient_displacement, cell, q)};
  // return compute_push_forward(S, F);

  AssertThrow(false,
              dealii::ExcMessage("This material does not (yet) support spatial integration."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::kirchhoff_stress(unsigned int const cell, unsigned int const q) const
  -> symmetric_tensor
{
  AssertThrow(false,
              dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                 "this material does not store tensorial quantities."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::contract_with_J_times_C(
  symmetric_tensor const & symmetric_gradient_increment,
  tensor const &           gradient_displacement,
  unsigned int const       cell,
  unsigned int const       q) const -> symmetric_tensor
{
  AssertThrow(false,
              dealii::ExcMessage("This material does not (yet) support spatial integration."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}


template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::contract_with_J_times_C(
  symmetric_tensor const & symmetric_gradient_increment,
  unsigned int const       cell,
  unsigned int const       q) const -> symmetric_tensor
{
  AssertThrow(false,
              dealii::ExcMessage("This function cannot be called with `non-caching` material."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}


template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::do_set_cell_linearization_data(
  std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
  unsigned int const                                                         cell) const -> void
{
  AssertThrow(false, dealii::ExcMessage("This material does not support caching."));

  return;
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::one_over_J(unsigned int const cell, unsigned int const q) const
  -> scalar
{
  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed one_over_J."));

  return dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN());
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::gradient_displacement(unsigned int const cell,
                                                   unsigned int const q) const -> tensor
{
  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed deformation gradient."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_tensor<dim, Number>());
}

template class AlveolarTissue<2, float>;
template class AlveolarTissue<3, float>;

template class AlveolarTissue<2, double>;
template class AlveolarTissue<3, double>;

} // namespace Structure
} // namespace ExaDG