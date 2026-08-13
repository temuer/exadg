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
#include <deal.II/matrix_free/matrix_free.h>
#include <exadg/structure/material/library/alveolar_tissue.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

using namespace ExaDG::Structure;

template<int dim>
struct DeformationCase
{
  std::array<std::array<double, dim>, dim> gradient;
  std::array<std::array<double, dim>, dim> increment;
  std::string                              name;
};

template<int dim, typename MaterialType>
bool
check_stress_increment(MaterialType &     material,
                       char const * const material_name,
                       bool const         forward_difference_for_undeformed_case = false)
{
  using tensor           = typename MaterialType::tensor;
  using symmetric_tensor = typename MaterialType::symmetric_tensor;
  using scalar           = typename MaterialType::scalar;

  constexpr int n_deformation_cases = 4;

  std::array<DeformationCase<dim>, 4> cases{};
  cases[0].name = "mixed deformation";
  cases[1].name = "mixed deformation from undeformed";
  cases[2].name = "pure stretch";
  cases[3].name = "shear deformation";

  cases[0].gradient  = {{{{0.18, 0.07, -0.02}}, {{-0.03, 0.11, 0.05}}, {{0.01, -0.04, 0.16}}}};
  cases[0].increment = {{{{0.04, -0.02, 0.01}}, {{0.03, -0.01, -0.02}}, {{-0.03, 0.02, 0.05}}}};
  cases[1].gradient  = {{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}}};
  cases[1].increment = {{{{0.04, -0.02, 0.01}}, {{0.03, -0.01, -0.02}}, {{-0.03, 0.02, 0.05}}}};
  cases[2].gradient  = {{{{0.12, 0.0, 0.0}}, {{0.0, -0.08, 0.0}}, {{0.0, 0.0, 0.16}}}};
  cases[2].increment = {{{{0.03, 0.0, 0.0}}, {{0.0, -0.02, 0.0}}, {{0.0, 0.0, 0.04}}}};
  cases[3].gradient  = {{{{0.0, 0.16, -0.03}}, {{-0.04, 0.0, 0.05}}, {{0.02, -0.06, 0.0}}}};
  cases[3].increment = {{{{0.01, -0.03, 0.02}}, {{0.02, -0.01, -0.02}}, {{-0.01, 0.03, 0.01}}}};

  tensor gradient{};
  tensor increment{};
  static_assert(n_deformation_cases == scalar::size());

  for(std::size_t lane{0}; lane < scalar::size(); ++lane)
  {
    for(int i = 0; i < dim; ++i)
      for(int j = 0; j < dim; ++j)
      {
        gradient[i][j][lane]  = cases[lane].gradient[i][j];
        increment[i][j][lane] = cases[lane].increment[i][j];
      }
  }

  symmetric_tensor const increment_analytical =
    material.second_piola_kirchhoff_stress_displacement_derivative(increment, gradient, 0, 0);

  std::array<double, 4> const step_sizes{{1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6}};
  for(double const epsilon : step_sizes)
  {
    tensor gradient_plus;
    tensor gradient_minus;
    for(int i = 0; i < dim; ++i)
      for(int j = 0; j < dim; ++j)
      {
        gradient_plus[i][j]  = gradient[i][j] + epsilon * increment[i][j];
        gradient_minus[i][j] = gradient[i][j] - epsilon * increment[i][j];
      }

    symmetric_tensor const stress_plus =
      material.second_piola_kirchhoff_stress(gradient_plus, 0, 0);
    symmetric_tensor const stress_minus =
      material.second_piola_kirchhoff_stress(gradient_minus, 0, 0);
    symmetric_tensor const stress_base = material.second_piola_kirchhoff_stress(gradient, 0, 0);

    for(std::size_t lane{0}; lane < scalar::size(); ++lane)
    {
      double max_error = 0.0;
      double max_scale = 1.0;

      bool const use_forward_difference = forward_difference_for_undeformed_case && lane == 1;

      for(int i = 0; i < dim; ++i)
        for(int j = 0; j <= i; ++j)
        {
          double const increment_finite_difference =
            use_forward_difference ?
              (stress_plus[i][j][lane] - stress_base[i][j][lane]) / epsilon :
              (stress_plus[i][j][lane] - stress_minus[i][j][lane]) / (2.0 * epsilon);
          max_error =
            std::max(max_error,
                     std::abs(increment_analytical[i][j][lane] - increment_finite_difference));
          max_scale = std::max(max_scale, std::abs(increment_finite_difference));
        }

      double const tolerance =
        1.0e-8 + (use_forward_difference ? 0.2 * epsilon : 1.0e-6) * max_scale;
      if(max_error > tolerance)
      {
        std::cerr << material_name << " (" << cases[lane].name << ", lane " << lane
                  << ") stress increment check failed at epsilon " << epsilon << " with error "
                  << max_error << " (tolerance " << tolerance << ")" << std::endl;
        return false;
      }
    }
  }

  return true;
}

int
main()
{
  WiechertSurfactantData surfactant_data;

  dealii::MatrixFree<3, double> matrix_free_3d;
  FibrousAlveolarTissueData<3>  fibrous_data_3d(
    MaterialType::Undefined, 1.2, 0.4, 1.7, 0.8, 2.3, Type2D::PlaneStrain, surfactant_data);
  NeoHookeAlveolarTissueData<3> neo_hooke_data_3d(
    MaterialType::Undefined, 2.0, 0.25, Type2D::PlaneStrain, surfactant_data);
  OgdenAlveolarTissueData<3> ogden_data_3d(
    MaterialType::Undefined, 1.2, 0.8, 1.7, 2.3, Type2D::PlaneStrain, surfactant_data);

  FibrousAlveolarTissue<3, double>  fibrous_3d(matrix_free_3d, 0, 0, fibrous_data_3d);
  NeoHookeAlveolarTissue<3, double> neo_hooke_3d(matrix_free_3d, 0, 0, neo_hooke_data_3d);
  OgdenAlveolarTissue<3, double>    ogden_3d(matrix_free_3d, 0, 0, ogden_data_3d);

  bool const passed = check_stress_increment<3>(fibrous_3d, "Fibrous", true) &&
                      check_stress_increment<3>(neo_hooke_3d, "Neo-Hooke") &&
                      check_stress_increment<3>(ogden_3d, "Ogden");

  if(passed)
    std::cout << "Alveolar tissue stress increment checks: PASS" << std::endl;

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
