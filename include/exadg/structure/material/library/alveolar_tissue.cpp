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
#include <algorithm>
#include <cmath>
#include <limits>

#include <exadg/structure/material/library/alveolar_tissue.h>

namespace ExaDG
{
namespace Structure
{

template<int dim, typename Number>
FibrousAlveolarTissue<dim, Number>::FibrousAlveolarTissue(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      dof_index,
  unsigned int const                      quad_index,
  FibrousAlveolarTissueData<dim> const &  data)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    surfactant_model(data.surfactant_data, matrix_free.n_boundary_face_batches())
{
}

template<int dim, typename Number>
auto
FibrousAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  return second_piola_kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number>
auto
FibrousAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_eval(
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
FibrousAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(unsigned int const cell,
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
FibrousAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
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
FibrousAlveolarTissue<dim, Number>::kirchhoff_stress(tensor const &     gradient_displacement,
                                                     unsigned int const cell,
                                                     unsigned int const q) const -> symmetric_tensor
{
  return kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number>
auto
FibrousAlveolarTissue<dim, Number>::kirchhoff_stress_eval(tensor const &     gradient_displacement,
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
FibrousAlveolarTissue<dim, Number>::kirchhoff_stress(unsigned int const cell,
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
FibrousAlveolarTissue<dim, Number>::contract_with_J_times_C(
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
FibrousAlveolarTissue<dim, Number>::contract_with_J_times_C(
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
FibrousAlveolarTissue<dim, Number>::do_set_cell_linearization_data(
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
FibrousAlveolarTissue<dim, Number>::one_over_J(unsigned int const cell,
                                               unsigned int const q) const -> scalar
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed one_over_J."));

  return dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN());
}

template<int dim, typename Number>
auto
FibrousAlveolarTissue<dim, Number>::gradient_displacement(unsigned int const cell,
                                                          unsigned int const q) const -> tensor
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed deformation gradient."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_tensor<dim, Number>());
}


template<int dim, typename Number>
NeoHookeAlveolarTissue<dim, Number>::NeoHookeAlveolarTissue(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      dof_index,
  unsigned int const                      quad_index,
  NeoHookeAlveolarTissueData<dim> const & data)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    surfactant_model(data.surfactant_data, matrix_free.n_boundary_face_batches())
{
}

template<int dim, typename Number>
auto
NeoHookeAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  return second_piola_kirchhoff_stress_eval(gradient_displacement, cell, q);
}

constexpr auto
get_mu(double const E, double const nu) -> double
{
  Assert(nu != -1.0, dealii::StandardExceptions::ExcDivideByZero());

  return 0.5 * E / (1.0 + nu);
}

constexpr auto
get_beta(double const nu) -> double
{
  Assert(nu != 0.5, dealii::StandardExceptions::ExcDivideByZero());

  return nu / (1.0 - 2.0 * nu);
}

template<int dim, typename Number>
auto
NeoHookeAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_eval(
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

  Number const mu   = static_cast<Number>(get_mu(data.E, data.nu));
  Number const beta = static_cast<Number>(get_beta(data.nu));

  scalar const J_pow = std::pow(J, static_cast<Number>(-2.0 * beta));

  symmetric_tensor const I_s = get_identity_symmetric_tensor<dim, Number>();

  return mu * (I_s - J_pow * C_inv);
}

template<int dim, typename Number>
auto
NeoHookeAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(unsigned int const cell,
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
NeoHookeAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
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

  Number const mu   = static_cast<Number>(get_mu(data.E, data.nu));
  Number const beta = static_cast<Number>(get_beta(data.nu));

  scalar const J_pow = std::pow(J, static_cast<Number>(-2.0 * beta));

  return mu * J_pow * (2.0 * beta * Du_J_over_J * C_inv - Du_C_inv);
}



template<int dim, typename Number>
auto
NeoHookeAlveolarTissue<dim, Number>::kirchhoff_stress(tensor const &     gradient_displacement,
                                                      unsigned int const cell,
                                                      unsigned int const q) const
  -> symmetric_tensor
{
  return kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number>
auto
NeoHookeAlveolarTissue<dim, Number>::kirchhoff_stress_eval(tensor const &     gradient_displacement,
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
NeoHookeAlveolarTissue<dim, Number>::kirchhoff_stress(unsigned int const cell, unsigned int const q)
  const -> symmetric_tensor
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
NeoHookeAlveolarTissue<dim, Number>::contract_with_J_times_C(
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
NeoHookeAlveolarTissue<dim, Number>::contract_with_J_times_C(
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
NeoHookeAlveolarTissue<dim, Number>::do_set_cell_linearization_data(
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
NeoHookeAlveolarTissue<dim, Number>::one_over_J(unsigned int const cell,
                                                unsigned int const q) const -> scalar
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed one_over_J."));

  return dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN());
}

template<int dim, typename Number>
auto
NeoHookeAlveolarTissue<dim, Number>::gradient_displacement(unsigned int const cell,
                                                           unsigned int const q) const -> tensor
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed deformation gradient."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_tensor<dim, Number>());
}

template<int dim, typename Number>
OgdenAlveolarTissue<dim, Number>::OgdenAlveolarTissue(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      dof_index,
  unsigned int const                      quad_index,
  OgdenAlveolarTissueData<dim> const &    data)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    surfactant_model(data.surfactant_data, matrix_free.n_boundary_face_batches())
{
}

template<int dim, typename Number>
auto
OgdenAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  return second_piola_kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number>
auto
OgdenAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_eval(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  (void)cell;
  (void)q;

  tensor const           F = dealii::Physics::Elasticity::Kinematics::F(gradient_displacement);
  symmetric_tensor const C = dealii::Physics::Elasticity::Kinematics::C(F);

  symmetric_tensor S{};

  // Serialize because eigenvectors() does not support VectorizedArray
  for(std::size_t v{0}; v < scalar::size(); ++v)
  {
    dealii::SymmetricTensor<2, dim, Number> C_lane;
    for(int i = 0; i < dim; ++i)
      for(int j = 0; j <= i; ++j)
        C_lane[i][j] = C[i][j][v];

    auto const eigen = dealii::eigenvectors(C_lane);

    for(int d = 0; d < dim; d++)
    {
      Number const principal_strain = std::sqrt(eigen[d].first);
      auto const & N                = eigen[d].second;

      Number const principal_stress = data.mu1 * std::pow(principal_strain, data.alpha1 - 2.0) +
                                      data.mu2 * std::pow(principal_strain, data.alpha2 - 2.0);

      for(int i = 0; i < dim; ++i)
        for(int j = 0; j <= i; ++j)
          S[i][j][v] += principal_stress * N[i] * N[j];
    }
  }

  return S;
}

template<int dim, typename Number>
auto
OgdenAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(unsigned int const cell,
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
OgdenAlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
  tensor const &     gradient_increment,
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  (void)cell;
  (void)q;

  tensor const           F    = dealii::Physics::Elasticity::Kinematics::F(gradient_displacement);
  symmetric_tensor const C    = dealii::Physics::Elasticity::Kinematics::C(F);
  symmetric_tensor const Du_C = dealii::symmetrize(dealii::transpose(gradient_increment) * F +
                                                   dealii::transpose(F) * gradient_increment);

  symmetric_tensor Du_S{};

  auto const h = [this](Number const rho) -> Number
  {
    return data.mu1 * std::pow(rho, 0.5 * (data.alpha1 - 2.0)) +
           data.mu2 * std::pow(rho, 0.5 * (data.alpha2 - 2.0));
  };

  auto const h_prime = [this](Number const rho) -> Number
  {
    return 0.5 * data.mu1 * (data.alpha1 - 2.0) * std::pow(rho, 0.5 * (data.alpha1 - 4.0)) +
           0.5 * data.mu2 * (data.alpha2 - 2.0) * std::pow(rho, 0.5 * (data.alpha2 - 4.0));
  };

  // Serialize because eigenvectors() does not support VectorizedArray.
  for(std::size_t v{0}; v < scalar::size(); ++v)
  {
    dealii::SymmetricTensor<2, dim, Number> C_lane;
    dealii::SymmetricTensor<2, dim, Number> Du_C_lane;
    for(int i = 0; i < dim; ++i)
      for(int j = 0; j <= i; ++j)
      {
        C_lane[i][j]    = C[i][j][v];
        Du_C_lane[i][j] = Du_C[i][j][v];
      }

    auto const eigen = dealii::eigenvectors(C_lane);

    for(int a = 0; a < dim; ++a)
    {
      Number const rho_a = eigen[a].first;
      auto const & N_a   = eigen[a].second;

      Number delta_rho_a = 0.0;
      for(int i = 0; i < dim; ++i)
        for(int j = 0; j < dim; ++j)
          delta_rho_a += N_a[i] * Du_C_lane[i][j] * N_a[j];

      Number const diagonal_coefficient = h_prime(rho_a) * delta_rho_a;
      for(int i = 0; i < dim; ++i)
        for(int j = 0; j <= i; ++j)
          Du_S[i][j][v] += diagonal_coefficient * N_a[i] * N_a[j];
    }

    for(int a = 0; a < dim; ++a)
      for(int b = a + 1; b < dim; ++b)
      {
        Number const rho_a = eigen[a].first;
        Number const rho_b = eigen[b].first;
        Number       divided_difference;

        Number const scale     = std::max(std::max(Number(1.0), std::abs(rho_a)), std::abs(rho_b));
        Number const tolerance = 100.0 * std::numeric_limits<Number>::epsilon() * scale;
        if(std::abs(rho_a - rho_b) <= tolerance)
          divided_difference = h_prime(rho_a);
        else
          divided_difference = (h(rho_a) - h(rho_b)) / (rho_a - rho_b);

        auto const & N_a = eigen[a].second;
        auto const & N_b = eigen[b].second;

        Number const mixed_increment = dealii::scalar_product(N_a, Du_C_lane * N_b);

        for(int i = 0; i < dim; ++i)
          for(int j = 0; j <= i; ++j)
            Du_S[i][j][v] +=
              divided_difference * mixed_increment * (N_a[i] * N_b[j] + N_b[i] * N_a[j]);
      }
  }

  return Du_S;
}



template<int dim, typename Number>
auto
OgdenAlveolarTissue<dim, Number>::kirchhoff_stress(tensor const &     gradient_displacement,
                                                   unsigned int const cell,
                                                   unsigned int const q) const -> symmetric_tensor
{
  return kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number>
auto
OgdenAlveolarTissue<dim, Number>::kirchhoff_stress_eval(tensor const &     gradient_displacement,
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
OgdenAlveolarTissue<dim, Number>::kirchhoff_stress(unsigned int const cell,
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
OgdenAlveolarTissue<dim, Number>::contract_with_J_times_C(
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
OgdenAlveolarTissue<dim, Number>::contract_with_J_times_C(
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
OgdenAlveolarTissue<dim, Number>::do_set_cell_linearization_data(
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
OgdenAlveolarTissue<dim, Number>::one_over_J(unsigned int const cell,
                                             unsigned int const q) const -> scalar
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed one_over_J."));

  return dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN());
}

template<int dim, typename Number>
auto
OgdenAlveolarTissue<dim, Number>::gradient_displacement(unsigned int const cell,
                                                        unsigned int const q) const -> tensor
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed deformation gradient."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_tensor<dim, Number>());
}
template class FibrousAlveolarTissue<2, float>;
template class FibrousAlveolarTissue<3, float>;

template class FibrousAlveolarTissue<2, double>;
template class FibrousAlveolarTissue<3, double>;

template class NeoHookeAlveolarTissue<2, float>;
template class NeoHookeAlveolarTissue<3, float>;

template class NeoHookeAlveolarTissue<2, double>;
template class NeoHookeAlveolarTissue<3, double>;

template class OgdenAlveolarTissue<2, float>;
template class OgdenAlveolarTissue<3, float>;

template class OgdenAlveolarTissue<2, double>;
template class OgdenAlveolarTissue<3, double>;

} // namespace Structure
} // namespace ExaDG
