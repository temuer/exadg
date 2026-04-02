/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef STRUCTURE_MATERIAL_LIBRARY_ALVEOLAR_TISSUE
#define STRUCTURE_MATERIAL_LIBRARY_ALVEOLAR_TISSUE

// deal.II
#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table.h>
#include <deal.II/base/types.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/variable_coefficients.h>
#include <exadg/structure/material/material.h>
#include "exadg/structure/user_interface/enum_types.h"

namespace ExaDG
{
namespace Structure
{
template<int dim>
struct AlveolarTissueData : public MaterialData
{
  using VectorType = dealii::LinearAlgebra::distributed::Vector<float>;

  AlveolarTissueData(MaterialType const & type) : MaterialData(type){};

  AlveolarTissueData(MaterialType const &                 type,
                     double const &                       shear_modulus,
                     double const &                       incompressibility_penalty,
                     double const &                       incompressibility_exponent,
                     double const &                       fiber_k_1,
                     double const &                       fiber_k_2,
                     std::set<dealii::types::boundary_id> surfactant_boundary_ids,
                     double const &                       surfactant_equilibrium_time,
                     double const &                       surfactant_m_1,
                     double const &                       surfactant_m_2,
                     double const &                       surfactant_k_1,
                     double const &                       surfactant_k_2,
                     double const &                       surfactant_c,
                     double const &                       relative_surfactant_concentration_max,
                     double const &                       surface_tension_ref,
                     double const &                       surface_tension_eq,
                     double const &                       surface_tension_min,
                     std::vector<unsigned int> const      degree_per_level,
                     double const &                       point_tolerance,
                     Type2D const &                       type_two_dim)
    : MaterialData(type),
      shear_modulus(shear_modulus),
      fiber_k_1(fiber_k_1),
      fiber_k_2(fiber_k_2),
      incompressibility_penalty(incompressibility_penalty),
      incompressibility_exponent(incompressibility_exponent),
      surfactant_boundary_ids(std::move(surfactant_boundary_ids)),
      surfactant_equilibrium_time(surfactant_equilibrium_time),
      surfactant_m_1(surfactant_m_1),
      surfactant_m_2(surfactant_m_2),
      surfactant_k_1(surfactant_k_1),
      surfactant_k_2(surfactant_k_2),
      surfactant_c(surfactant_c),
      relative_surfactant_concentration_max(relative_surfactant_concentration_max),
      surface_tension_ref(surface_tension_ref),
      surface_tension_eq(surface_tension_eq),
      surface_tension_min(surface_tension_min),
      degree_per_level(std::move(degree_per_level)),
      point_tolerance(point_tolerance),
      type_two_dim(type_two_dim)
  {
    AssertThrow(surfactant_equilibrium_time > 0.0,
                dealii::ExcMessage("equilibrium time must be greater than 0"));
  }

  // Ground substance
  double shear_modulus{0.0};

  // Fiber
  double fiber_k_1{0.0};
  double fiber_k_2{0.0};

  // Incompressibility
  double incompressibility_penalty{0.0};
  double incompressibility_exponent{1.0};

  // Surfactant

  std::set<dealii::types::boundary_id> surfactant_boundary_ids{};
  double                               surfactant_equilibrium_time{0.0};
  double                               surfactant_m_1{0.0};
  double                               surfactant_m_2{0.0};
  double                               surfactant_k_1{0.0};
  double                               surfactant_k_2{0.0};
  double                               surfactant_c{0.0};
  double                               relative_surfactant_concentration_max{1.0};
  double                               surface_tension_ref{0.0};
  double                               surface_tension_eq{0.0};
  double                               surface_tension_min{0.0};

  std::vector<unsigned int> degree_per_level{};

  double point_tolerance{0.0};

  Type2D type_two_dim{Type2D::Undefined};
};

/*
 * Psi_gs = shear_modulus / 2 * ( I_1 * J^(-2/3) - dim )
 *
 * Psi_pen = penalty * ( J^(2*exponent) + J^(-2*exponent) - 2), penalty > 0, exponent > 1
 *
 * Psi_fib = | k_1 / (2 * k_2) * exp(k_2 * ((I_1 / 3) - 1)^2 - 1) if I_1 >= 3
 *           | 0 else
 *
 * Psi = Psi_gs + Psi_pen + Psi_fib
 */
template<int dim, typename Number>
class AlveolarTissue : public Material<dim, Number>
{
public:
  using VectorType     = dealii::LinearAlgebra::distributed::Vector<Number>;
  using Range          = std::pair<unsigned int, unsigned int>;
  using IntegratorCell = CellIntegrator<dim, dim, Number>;

  using scalar           = dealii::VectorizedArray<Number>;
  using vector           = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  using tensor           = dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>;
  using symmetric_tensor = dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>;

  AlveolarTissue(dealii::MatrixFree<dim, Number> const & matrix_free,
                 unsigned int const                      dof_index,
                 unsigned int const                      quad_index,
                 AlveolarTissueData<dim> const &         data);

  std::pair<scalar, std::array<unsigned int, scalar::size()>>
  surface_tension(scalar const &     surface_area_new,
                  double const       time_step_size,
                  unsigned int const boundary_face) const;

  scalar
  surface_tension_increment(scalar const & surface_area_new,
                            scalar const & surface_area_new_increment,
                            std::array<unsigned int, scalar::size()> const & regime,
                            double const                                     time_step_size,
                            unsigned int const                               boundary_face) const;

  void
  update_material(scalar const &     surface_area_new,
                  double const       time,
                  double const       time_step_size,
                  unsigned int const boundary_face);

  /*
   * S_gs  = shear_modulus * J^(-2/3) * (I - I_1 / 3 * C^(-1) )
   *
   * S_pen = 2 * penalty * exponent * (J^(2*exponent) - J^(-2*exponent) ) C^(-1)
   *
   * S_fib = | 2 * k_1 / 3 * (I_1 / 3 - 1) * exp(k_2 * (I_1 / 3 - 1)^2) * I if I_1 >= 3
   *         | 0 else
   *
   * S = S_gs + S_pen + S_fib
   */
  symmetric_tensor
  second_piola_kirchhoff_stress(tensor const &     gradient_displacement,
                                unsigned int const cell,
                                unsigned int const q) const final;

  symmetric_tensor
  second_piola_kirchhoff_stress_eval(tensor const &     gradient_displacement,
                                     unsigned int const cell,
                                     unsigned int const q) const final;

  symmetric_tensor
  second_piola_kirchhoff_stress(unsigned int const cell, unsigned int const q) const final;

  /*
   * Stress increment Du_S
   */
  symmetric_tensor
  second_piola_kirchhoff_stress_displacement_derivative(tensor const &     gradient_increment,
                                                        tensor const &     gradient_displacement,
                                                        unsigned int const cell,
                                                        unsigned int const q) const final;
  /*
   * Surface tension stress as 1PK
   */
  tensor
  surface_tension_1PK(tensor const &     gradient_displacement,
                      vector const &     material_normal_vector,
                      scalar const &     surface_area_new,
                      double const       time,
                      double const       time_step_size,
                      unsigned int const face) const;

  /*
   * Surface tension stress increment Du_P_gamma
   */
  tensor
  surface_tension_1PK_displacement_derivative(tensor const &     displacement_gradient_increment,
                                              tensor const &     displacement_gradient,
                                              vector const &     material_normal_vector,
                                              scalar const &     surface_area_new,
                                              scalar const &     surface_area_new_increment,
                                              double const       time,
                                              double const       time_step_size,
                                              unsigned int const face) const;

  symmetric_tensor
  kirchhoff_stress(tensor const &     gradient_displacement,
                   unsigned int const cell,
                   unsigned int const q) const final;

  symmetric_tensor
  kirchhoff_stress_eval(tensor const &     gradient_displacement,
                        unsigned int const cell,
                        unsigned int const q) const final;

  symmetric_tensor
  kirchhoff_stress(unsigned int const cell, unsigned int const q) const final;

  symmetric_tensor
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          tensor const &           gradient_displacement,
                          unsigned int const       cell,
                          unsigned int const       q) const final;

  symmetric_tensor
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          unsigned int const       cell,
                          unsigned int const       q) const final;

  void
  do_set_cell_linearization_data(
    std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
    unsigned int const                                                         cell) const final;

  scalar
  one_over_J(unsigned int const cell, unsigned int const q) const final;

  tensor
  gradient_displacement(unsigned int const cell, unsigned int const q) const final;

  bool
  is_surfactant_boundary(dealii::types::boundary_id boundary_id) const
  {
    return data.surfactant_boundary_ids.find(boundary_id) != data.surfactant_boundary_ids.end();
  }

private:
  unsigned int dof_index;
  unsigned int quad_index;

  AlveolarTissueData<dim> const & data;

  struct SurfactantModel
  {
    scalar relative_concentration;
    scalar surface_area;
  };

  dealii::Table<1, SurfactantModel> surfactant_internal_variables_old;
};

} // namespace Structure
} // namespace ExaDG

#endif /* STRUCTURE_MATERIAL_LIBRARY_INCOMPRESSIBLE_FIBROUS_TISSUE */
