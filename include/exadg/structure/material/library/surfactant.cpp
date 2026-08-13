/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2026 by the ExaDG authors
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
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>
#include <cmath>


#include <exadg/structure/material/library/surfactant.h>

namespace ExaDG
{
namespace Structure
{

template<int dim, typename Number>
WiechertSurfactantModel<dim, Number>::WiechertSurfactantModel(WiechertSurfactantData const & data,
                                                              unsigned int n_boundary_face_batches)
  : data(data), variables_old(n_boundary_face_batches, SurfactantVariables{1.0, 1.0})
{
}

template<int dim, typename Number>
bool
WiechertSurfactantModel<dim, Number>::is_surfactant_boundary(
  dealii::types::boundary_id boundary_id) const
{
  return data.boundary_ids.find(boundary_id) != data.boundary_ids.end();
}

template<int dim, typename Number>
auto
WiechertSurfactantModel<dim, Number>::surface_tension_1PK(tensor const &     displacement_gradient,
                                                          vector const &     material_normal_vector,
                                                          scalar const &     surface_area_new,
                                                          double const       time,
                                                          double const       time_step_size,
                                                          unsigned int const face) const -> tensor
{
  //! UNTIL EQUIILIBRIUM TIME IS ACHIEVED, GRADUALLY APPLY EQUILIBRIUM SURFACE TENSION
  auto const [gamma, regime] =
    time < data.equilibrium_time ?
      std::pair<scalar, std::array<unsigned int, scalar::size()>>{
        dealii::make_vectorized_array<Number>(data.gamma_eq * time / data.equilibrium_time), {}} :
      surface_tension_and_current_regime(surface_area_new, time_step_size, face);

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
WiechertSurfactantModel<dim, Number>::surface_tension_1PK_displacement_derivative(
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
  auto const [gamma, regime] =
    time < data.equilibrium_time ?
      std::pair<scalar, std::array<unsigned int, scalar::size()>>{
        dealii::make_vectorized_array<Number>(0.0), {}} :
      surface_tension_and_current_regime(surface_area_new, time_step_size, face);

  scalar const Du_gamma = time < data.equilibrium_time ?
                            0.0 :
                            surface_tension_displacement_derivative(surface_area_new,
                                                                    surface_area_new_increment,
                                                                    regime,
                                                                    time_step_size,
                                                                    face);

  // TODO! Clear what this is
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
WiechertSurfactantModel<dim, Number>::surface_tension_and_current_regime(
  scalar const &     surface_area_new,
  double const       time_step_size,
  unsigned int const face) const -> std::pair<scalar, std::array<unsigned int, scalar::size()>>
{
  scalar gamma = dealii::make_vectorized_array<Number>(0.0);

  std::array<unsigned int, scalar::size()> regime{};

  scalar const & relative_concentration_old = variables_old[face].relative_concentration;
  scalar const & surface_area_old           = variables_old[face].surface_area;

  for(std::size_t v{0}; v < scalar::size(); v++)
  {
    if(relative_concentration_old[v] < 1.0)
    // Regime 1
    {
      Number const inv_delta_t = 1.0 / time_step_size;
      Number const k1_C        = data.k_1 * data.c;
      Number const k2          = data.k_2;

      Number const relative_concentration_new =
        ((relative_concentration_old[v] * surface_area_old[v] * inv_delta_t) +
         (surface_area_new[v] * k1_C)) /
        (surface_area_new[v] * (inv_delta_t + k1_C + k2));

      gamma[v]  = data.gamma_ref - data.m_1 * relative_concentration_new;
      regime[v] = 1;
    }
    else if(relative_concentration_old[v] < data.relative_concentration_max)
    // Regime 2
    {
      Number const relative_concentration_new =
        relative_concentration_old[v] * surface_area_old[v] / surface_area_new[v];

      gamma[v]  = data.gamma_eq - data.m_2 * (relative_concentration_new - 1.0);
      regime[v] = 2;
    }
    else
    // Regime 3
    {
      gamma[v]  = data.gamma_min;
      regime[v] = 3;
    }
  }

  return {gamma, regime};
}

template<int dim, typename Number>
auto
WiechertSurfactantModel<dim, Number>::surface_tension_displacement_derivative(
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
      Number const relative_concentration_old = variables_old[face_id].relative_concentration[v];
      Number const surface_area_old           = variables_old[face_id].surface_area[v];

      Du_gamma[v] = static_cast<Number>(data.m_1) * relative_concentration_old * surface_area_old /
                    (time_step_size * surface_area_new[v] * surface_area_new[v] *
                     (1.0 / time_step_size + data.k_1 * data.c + data.k_2)) *
                    surface_area_new_increment[v];
    }
    else if(regime[v] == 2)
    {
      Number const relative_concentration_old = variables_old[face_id].relative_concentration[v];
      Number const surface_area_old           = variables_old[face_id].surface_area[v];

      Du_gamma[v] = static_cast<Number>(data.m_2) * relative_concentration_old * surface_area_old /
                    (surface_area_new[v] * surface_area_new[v]) * surface_area_new_increment[v];
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
WiechertSurfactantModel<dim, Number>::update(scalar const &     surface_area_new,
                                             double const       time,
                                             double const       time_step_size,
                                             unsigned int const boundary_face_id) -> void
{
  //! UNTIL EQUILIBRIUM TIME IS ACHIEVED
  if(time < data.equilibrium_time)
  {
    // Until time is queal to equilibrium time, history variables are not used. For later use, set
    // them to the equilibrium state.
    variables_old[boundary_face_id].relative_concentration = 1.0;

    // Update surface area
    variables_old[boundary_face_id].surface_area = surface_area_new;
  }

  scalar const relative_concentration_old = variables_old[boundary_face_id].relative_concentration;
  scalar const surface_area_old           = variables_old[boundary_face_id].surface_area;

  for(std::size_t v{0}; v < scalar::size(); v++)
  {
    Number relative_concentration_new;
    if(relative_concentration_old[v] < 1.0)
    // Regime 1
    {
      Number const inv_delta_t = 1.0 / time_step_size;
      Number const k1_C        = data.k_1 * data.c;
      Number const k2          = data.k_2;

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
    variables_old[boundary_face_id].relative_concentration[v] =
      std::min(relative_concentration_new, static_cast<Number>(data.relative_concentration_max));
  }

  // Update surface area
  variables_old[boundary_face_id].surface_area = surface_area_new;
}

template class WiechertSurfactantModel<2, double>;
template class WiechertSurfactantModel<3, double>;

template class WiechertSurfactantModel<2, float>;
template class WiechertSurfactantModel<3, float>;

} // namespace Structure

} // namespace ExaDG