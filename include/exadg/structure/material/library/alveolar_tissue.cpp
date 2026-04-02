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
  surfactant_internal_variables_old.reinit(matrix_free.n_boundary_face_batches());

  surfactant_internal_variables_old.fill(SurfactantModel{1.0, 1.0});
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::surface_tension(scalar const &     surface_area_new,
                                             double const       time_step_size,
                                             unsigned int const face) const
  -> std::pair<scalar, std::array<unsigned int, scalar::size()>>
{
  scalar                                   gamma = dealii::make_vectorized_array<Number>(0.0);
  std::array<unsigned int, scalar::size()> regime{};

  SurfactantModel const & internal_variables_old{surfactant_internal_variables_old[face]};

  scalar const & relative_concentration_old = internal_variables_old.relative_concentration;
  scalar const & surface_area_old           = internal_variables_old.surface_area;

  for(std::size_t v{0}; v < scalar::size(); v++)
  {
    if(relative_concentration_old[v] < 1.0)
    // Regime 1
    {
      Number const inv_delta_t = 1.0 / time_step_size;
      Number const k1_C        = data.surfactant_k_1 * data.surfactant_c;
      Number const k2          = data.surfactant_k_2;

      Number const relative_concentration_new =
        ((relative_concentration_old[v] * surface_area_old[v] * inv_delta_t) +
         (surface_area_new[v] * k1_C)) /
        (surface_area_new[v] * (inv_delta_t + k1_C + k2));

      gamma[v]  = data.surface_tension_ref - data.surfactant_m_1 * relative_concentration_new;
      regime[v] = 1;
    }
    else if(relative_concentration_old[v] < data.relative_surfactant_concentration_max)
    // Regime 2
    {
      Number const relative_concentration_new =
        relative_concentration_old[v] * surface_area_old[v] / surface_area_new[v];

      gamma[v] = data.surface_tension_eq - data.surfactant_m_2 * (relative_concentration_new - 1.0);
      regime[v] = 2;
    }
    else
    // Regime 3
    {
      gamma[v]  = data.surface_tension_min;
      regime[v] = 3;
    }
  }

  return {gamma, regime};
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::surface_tension_increment(
  scalar const &                                   surface_area_new,
  scalar const &                                   surface_area_new_increment,
  std::array<unsigned int, scalar::size()> const & regime,
  double const                                     time_step_size,
  unsigned int const                               face_id) const -> scalar
{
  scalar Du_gamma = dealii::make_vectorized_array<Number>(0.0);

  for(std::size_t v{0}; v < scalar::size(); v++)
  {
    if(regime[v] == 1)
    {
      Number const relative_concentration_old =
        surfactant_internal_variables_old[face_id].relative_concentration[v];
      Number const surface_area_old = surfactant_internal_variables_old[face_id].surface_area[v];

      Du_gamma[v] =
        static_cast<Number>(data.surfactant_m_1) * relative_concentration_old * surface_area_old /
        (time_step_size * surface_area_new[v] * surface_area_new[v] *
         (1.0 / time_step_size + data.surfactant_k_1 * data.surfactant_c + data.surfactant_k_2)) *
        surface_area_new_increment[v];
    }
    else if(regime[v] == 2)
    {
      Number const relative_concentration_old =
        surfactant_internal_variables_old[face_id].relative_concentration[v];
      Number const surface_area_old = surfactant_internal_variables_old[face_id].surface_area[v];

      Du_gamma[v] = static_cast<Number>(data.surfactant_m_2) * relative_concentration_old *
                    surface_area_old / (surface_area_new[v] * surface_area_new[v]) *
                    surface_area_new_increment[v];
    }
    else if(regime[v] == 3)
    {
      Du_gamma[v] = 0.0;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("invalid surfactant regime"));
    }
  }

  return Du_gamma;
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::update_material(scalar const &     surface_area_new,
                                             double const       time,
                                             double const       time_step_size,
                                             unsigned int const boundary_face_id) -> void
{
  //! UNTIL EQUILIBRIUM TIME IS ACHIEVED
  if(time < data.surfactant_equilibrium_time)
  {
    // Until time is queal to equilibrium time, history variables are not used. For later use, set
    // them to the equilibrium state.
    surfactant_internal_variables_old[boundary_face_id].relative_concentration = 1.0;

    // Update surface area
    surfactant_internal_variables_old[boundary_face_id].surface_area = surface_area_new;
  }

  scalar const relative_concentration_old =
    surfactant_internal_variables_old[boundary_face_id].relative_concentration;
  scalar const surface_area_old = surfactant_internal_variables_old[boundary_face_id].surface_area;

  for(std::size_t v{0}; v < scalar::size(); v++)
  {
    Number relative_concentration_new;
    if(relative_concentration_old[v] < 1.0)
    // Regime 1
    {
      Number const inv_delta_t = 1.0 / time_step_size;
      Number const k1_C        = data.surfactant_k_1 * data.surfactant_c;
      Number const k2          = data.surfactant_k_2;

      relative_concentration_new =
        ((relative_concentration_old[v] * surface_area_old[v] * inv_delta_t) +
         (surface_area_new[v] * k1_C)) /
        (surface_area_new[v] * (inv_delta_t + k1_C + k2));
    }
    else
    // Regime 2 & 3
    {
      relative_concentration_new =
        relative_concentration_old[v] * surface_area_old[v] / surface_area_new[v];
    }

    // Update relative concentration
    surfactant_internal_variables_old[boundary_face_id].relative_concentration[v] =
      std::min(relative_concentration_new,
               static_cast<Number>(data.relative_surfactant_concentration_max));
  }

  // Update surface area
  surfactant_internal_variables_old[boundary_face_id].surface_area = surface_area_new;
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
AlveolarTissue<dim, Number>::second_piola_kirchhoff_stress(unsigned int const cell,
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
AlveolarTissue<dim, Number>::second_piola_kirchhoff_stress_displacement_derivative(
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
AlveolarTissue<dim, Number>::surface_tension_1PK(tensor const &     displacement_gradient,
                                                 vector const &     material_normal_vector,
                                                 scalar const &     surface_area_new,
                                                 double const       time,
                                                 double const       time_step_size,
                                                 unsigned int const face) const -> tensor
{
  //! UNTIL EQUIILIBRIUM TIME IS ACHIEVED, GRADUALLY APPLY EQUILIBRIUM SURFACE TENSION
  auto const [gamma, regime] =
    time < data.surfactant_equilibrium_time ?
      std::pair<scalar, std::array<unsigned int, scalar::size()>>{
        dealii::make_vectorized_array<Number>(data.surface_tension_eq * time /
                                              data.surfactant_equilibrium_time),
        {}} :
      surface_tension(surface_area_new, time_step_size, face);

  // Regime is not needed further here.
  (void)regime;

  tensor const           F           = compute_F(displacement_gradient);
  tensor const           F_inv       = dealii::invert(F);
  scalar const           J           = dealii::determinant(F);
  vector const           n_star      = J * dealii::transpose(F_inv) * material_normal_vector;
  scalar const           n_star_norm = n_star.norm();
  vector const           n           = n_star / n_star_norm;
  symmetric_tensor const n_projector =
    get_identity_symmetric_tensor<dim, Number>() - dealii::symmetrize(dealii::outer_product(n, n));


  return gamma * n_star_norm * n_projector * dealii::transpose(F_inv);
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::surface_tension_1PK_displacement_derivative(
  tensor const &     displacement_gradient_increment,
  tensor const &     displacement_gradient,
  vector const &     material_normal_vector,
  scalar const &     surface_area_new,
  scalar const &     surface_area_new_increment,
  double const       time,
  double const       time_step_size,
  unsigned int const face) const -> tensor
{
  //! UNTIL EQUIILIBRIUM TIME IS ACHIEVED, GRADUALLY APPLY EQUILIBRIUM SURFACE TENSION
  auto const [gamma, regime] = time < data.surfactant_equilibrium_time ?
                                 std::pair<scalar, std::array<unsigned int, scalar::size()>>{
                                   dealii::make_vectorized_array<Number>(0.0), {}} :
                                 surface_tension(surface_area_new, time_step_size, face);

  scalar const Du_gamma = time < data.surfactant_equilibrium_time ?
                            0.0 :
                            surface_tension_increment(surface_area_new,
                                                      surface_area_new_increment,
                                                      regime,
                                                      time_step_size,
                                                      face);
  (void)Du_gamma;

  tensor const           F     = dealii::Physics::Elasticity::Kinematics::F(displacement_gradient);
  tensor const           F_inv = dealii::invert(F);
  tensor const           F_inv_T     = dealii::transpose(F_inv);
  scalar const           J           = dealii::determinant(F);
  vector const           n_star      = J * material_normal_vector * F_inv;
  scalar const           n_star_norm = n_star.norm();
  vector const           n           = n_star / n_star_norm;
  symmetric_tensor const n_projector =
    get_identity_symmetric_tensor<dim, Number>() - dealii::symmetrize(dealii::outer_product(n, n));

  tensor const Du_F_inv = -F_inv * displacement_gradient_increment * F_inv;
  scalar const Du_J     = J * dealii::trace(displacement_gradient_increment * F_inv);

  vector const Du_n_star =
    Du_J * material_normal_vector * F_inv + J * material_normal_vector * Du_F_inv;
  scalar const Du_n_star_norm = n * Du_n_star;

  tensor Du_P_gamma{};

  // Du gamma term
  {
    Du_P_gamma += Du_gamma * n_star_norm * n_projector * F_inv_T;
  }

  // Du n_star_norm term
  {
    Du_P_gamma += gamma * Du_n_star_norm * n_projector * F_inv_T;
  }

  // Du n_projector term
  {
    vector           Du_n           = (1.0 / n_star_norm) * n_projector * Du_n_star;
    symmetric_tensor Du_n_projector = 2.0 * dealii::symmetrize(dealii::outer_product(n, Du_n));
    Du_P_gamma += -gamma * n_star_norm * Du_n_projector * F_inv_T;
  }

  // Du F_inv_T term
  {
    Du_P_gamma += gamma * n_star_norm * n_projector * dealii::transpose(Du_F_inv);
  }

  return Du_P_gamma;
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
  (void)gradient_displacement;
  (void)cell;
  (void)q;

  AssertThrow(false,
              dealii::ExcMessage("This material does not (yet) support spatial integration."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_symmetric_tensor<dim, Number>());
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::kirchhoff_stress(unsigned int const cell, unsigned int const q) const
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
AlveolarTissue<dim, Number>::contract_with_J_times_C(
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
AlveolarTissue<dim, Number>::contract_with_J_times_C(
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
AlveolarTissue<dim, Number>::do_set_cell_linearization_data(
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
AlveolarTissue<dim, Number>::one_over_J(unsigned int const cell, unsigned int const q) const
  -> scalar
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed one_over_J."));

  return dealii::make_vectorized_array(std::numeric_limits<Number>::quiet_NaN());
}

template<int dim, typename Number>
auto
AlveolarTissue<dim, Number>::gradient_displacement(unsigned int const cell,
                                                   unsigned int const q) const -> tensor
{
  (void)cell;
  (void)q;

  AssertThrow(false, dealii::ExcMessage("Cannot access precomputed deformation gradient."));

  return (std::numeric_limits<Number>::quiet_NaN() * get_identity_tensor<dim, Number>());
}

template class AlveolarTissue<2, float>;
template class AlveolarTissue<3, float>;

template class AlveolarTissue<2, double>;
template class AlveolarTissue<3, double>;

} // namespace Structure
} // namespace ExaDG