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
#include <limits>
#include <numeric>

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
FibrousAlveolarTissue<dim, Number>::one_over_J(unsigned int const cell, unsigned int const q) const
  -> scalar
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

  auto const Du_S = mu * J_pow * (2.0 * beta * Du_J_over_J * C_inv - Du_C_inv);

  return Du_S;
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
NeoHookeAlveolarTissue<dim, Number>::kirchhoff_stress(unsigned int const cell,
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
NeoHookeAlveolarTissue<dim, Number>::one_over_J(unsigned int const cell, unsigned int const q) const
  -> scalar
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

template<int dim, typename Number, int cache_level>
OgdenAlveolarTissue<dim, Number, cache_level>::OgdenAlveolarTissue(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      dof_index,
  unsigned int const                      quad_index,
  OgdenAlveolarTissueData<dim> const &    data)
  : dof_index(dof_index),
    quad_index(quad_index),
    data(data),
    surfactant_model(data.surfactant_data, matrix_free.n_boundary_face_batches())
{
  if constexpr(cache_level == 0)
  {
    return;
  }

  eigendecomposition_coefficients.initialize(matrix_free, quad_index, false, false);
  eigendecomposition_coefficients.set_coefficients(OgdenEigendecomposition<dim, Number>{});
}

template<int dim, typename Number>
auto
outer_product_self(dealii::Tensor<1, dim, Number> const & vec)
  -> dealii::SymmetricTensor<2, dim, Number>
{
  dealii::SymmetricTensor<2, dim, Number> res{};

  for(int i = 0; i < dim; ++i)
  {
    for(int j = i; j < dim; ++j)
    {
      res[i][j] = vec[i] * vec[j];
    }
  }

  return res;
}

template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::second_piola_kirchhoff_stress(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  static_assert(cache_level == 0 || cache_level == 1, "Only cache_level 0 or 1 implemented");

  if constexpr(cache_level == 0)
  {
    return second_piola_kirchhoff_stress_eval(gradient_displacement, cell, q);
  }

  tensor const           F = dealii::Physics::Elasticity::Kinematics::F(gradient_displacement);
  scalar const           J = dealii::determinant(F);
  symmetric_tensor const C = dealii::Physics::Elasticity::Kinematics::C(F);

  // Isochoric stress
  symmetric_tensor const S_iso = std::invoke(
    [&]()
    {
      OgdenEigendecomposition<dim, Number> const & cache =
        eigendecomposition_coefficients.get_coefficient_cell(cell, q);

      symmetric_tensor S_iso{};
      for(int a = 0; a < dim; a++)
      {
        S_iso += cache.S_iso[a] * outer_product_self(cache.eigenvectors[a]);
      }

      return S_iso;
    });

  // Volumetric stress
  symmetric_tensor const S_vol = std::invoke(
    [&]()
    {
      symmetric_tensor const C_inv = dealii::invert(C);
      return 0.5 * data.kappa * (J * J - 1.0) * C_inv;
    });

  return S_iso + S_vol;
}



template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::second_piola_kirchhoff_stress_eval(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  (void)cell;
  (void)q;

  tensor const           F = dealii::Physics::Elasticity::Kinematics::F(gradient_displacement);
  scalar const           J = dealii::determinant(F);
  symmetric_tensor const C = dealii::Physics::Elasticity::Kinematics::C(F);

  // Isochoric stress
  symmetric_tensor const S_iso = std::invoke(
    [&]()
    {
      auto const [lambda_sq, N] = compute_eigenbasis(C);

      scalar const J_pow = std::pow(J, static_cast<Number>(-ONE_THIRD));

      std::array<scalar, static_cast<size_t>(dim)> principal_stresses =
        principal_isochoric_2PK_stresses<dim>(lambda_sq,
                                              J_pow,
                                              static_cast<Number>(data.mu1),
                                              static_cast<Number>(data.mu2),
                                              static_cast<Number>(data.alpha1),
                                              static_cast<Number>(data.alpha2));

      symmetric_tensor S_iso{};
      for(int a = 0; a < dim; a++)
      {
        S_iso += principal_stresses[a] * outer_product_self(N[a]);
      }

      return S_iso;
    });

  // Volumetric stress
  symmetric_tensor const S_vol = std::invoke(
    [&]()
    {
      symmetric_tensor const C_inv = dealii::invert(C);
      return 0.5 * data.kappa * (J * J - 1.0) * C_inv;
    });

  return S_iso + S_vol;
}

template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::second_piola_kirchhoff_stress(
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This function implements loading a stored stress tensor, but "
                                 "this material does not store tensorial quantities."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::
  second_piola_kirchhoff_stress_displacement_derivative(tensor const &     gradient_increment,
                                                        tensor const &     gradient_displacement,
                                                        unsigned int const cell,
                                                        unsigned int const q) const
  -> symmetric_tensor
{
  static_assert(cache_level == 0 || cache_level == 1, "Only cache_level 0 or 1 implemented.");

  tensor const F = dealii::Physics::Elasticity::Kinematics::F(gradient_displacement);
  scalar const J = dealii::determinant(F);

  symmetric_tensor const C     = dealii::Physics::Elasticity::Kinematics::C(F);
  symmetric_tensor const C_inv = dealii::invert(C);

  symmetric_tensor const Du_C = 2.0 * dealii::symmetrize(dealii::transpose(gradient_increment) * F);

  // Cached or not-cached values
  OgdenEigendecomposition<dim, Number> const deformation = std::invoke(
    [this](symmetric_tensor const & C, scalar const & J, unsigned int cell, unsigned int q)
    {
      if constexpr(cache_level)
      {
        return eigendecomposition_coefficients.get_coefficient_cell(cell, q);
      }
      else
      {
        return ogden_eigendecomposition(C,
                                        J,
                                        static_cast<Number>(data.mu1),
                                        static_cast<Number>(data.mu2),
                                        static_cast<Number>(data.alpha1),
                                        static_cast<Number>(data.alpha2));
      }
    },
    C,
    J,
    cell,
    q);

  // Cache dS_dlambda_over_lamda as symmetric_tensor
  // Cache fac as tensor
  // Write two loops (one with full tensor, second only non-diag part)
  symmetric_tensor Du_S{};
  for(int a = 0; a < dim; ++a)
  {
    for(int b = 0; b < dim; ++b)
    {
      vector const Du_C_Nb = Du_C * deformation.eigenvectors[b];
      scalar const g_b     = deformation.eigenvectors[b] * Du_C_Nb;

      Du_S += (0.5 * deformation.c1[a][b] * g_b) * outer_product_self(deformation.eigenvectors[a]);

      if(a == b)
        continue;

      scalar const h_ab = deformation.eigenvectors[a] * Du_C_Nb;

      Du_S += (deformation.c2[a][b] * h_ab) *
              dealii::symmetrize(
                dealii::outer_product(deformation.eigenvectors[a], deformation.eigenvectors[b]));
    }
  }

  // Volumetric part
  tensor const           F_inv       = dealii::invert(F);
  tensor const           Du_F_inv    = -F_inv * gradient_increment * F_inv;
  symmetric_tensor const Du_C_inv    = compute_H_plus_HT(Du_F_inv * dealii::transpose(F_inv));
  scalar const           Du_J_over_J = dealii::trace(gradient_increment * F_inv);

  // J term
  Du_S += data.kappa * J * J * Du_J_over_J * C_inv;

  // C_inv term
  Du_S += 0.5 * data.kappa * (J * J - 1.0) * Du_C_inv;

  return Du_S;
}

template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::kirchhoff_stress(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  return kirchhoff_stress_eval(gradient_displacement, cell, q);
}

template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::kirchhoff_stress_eval(
  tensor const &     gradient_displacement,
  unsigned int const cell,
  unsigned int const q) const -> symmetric_tensor
{
  (void)gradient_displacement;
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This material does not (yet) support spatial integration."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::kirchhoff_stress(unsigned int const cell,
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

template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::contract_with_J_times_C(
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

template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::contract_with_J_times_C(
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

template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::do_set_cell_linearization_data(
  std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
  unsigned int const                                                         cell) const -> void
{
  static_assert(cache_level < 2, "Only cache_level 0 and 1 are implemented.");

  AssertThrow(cache_level > 0, dealii::ExcMessage("Cache 0 should not arrive here."));

  for(unsigned int q = 0; q < integrator_lin->n_q_points; ++q)
  {
    tensor const Grad_d_lin = integrator_lin->get_gradient(q);

    tensor const           F = dealii::Physics::Elasticity::Kinematics::F(Grad_d_lin);
    scalar const           J = dealii::determinant(F);
    symmetric_tensor const C = dealii::Physics::Elasticity::Kinematics::C(F);

    eigendecomposition_coefficients.set_coefficient_cell(
      cell,
      q,
      ogden_eigendecomposition(C,
                               J,
                               static_cast<Number>(data.mu1),
                               static_cast<Number>(data.mu2),
                               static_cast<Number>(data.alpha1),
                               static_cast<Number>(data.alpha2)));
  }

  return;
}

template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::one_over_J(unsigned int const cell,
                                                          unsigned int const q) const -> scalar
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed one_over_J."));

  return dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN());
}

template<int dim, typename Number, int cache_level>
auto
OgdenAlveolarTissue<dim, Number, cache_level>::gradient_displacement(unsigned int const cell,
                                                                     unsigned int const q) const
  -> tensor
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

template class OgdenAlveolarTissue<2, float, 0>;
template class OgdenAlveolarTissue<3, float, 0>;

template class OgdenAlveolarTissue<2, double, 0>;
template class OgdenAlveolarTissue<3, double, 0>;

template class OgdenAlveolarTissue<2, float, 1>;
template class OgdenAlveolarTissue<3, float, 1>;

template class OgdenAlveolarTissue<2, double, 1>;
template class OgdenAlveolarTissue<3, double, 1>;
} // namespace Structure
} // namespace ExaDG
