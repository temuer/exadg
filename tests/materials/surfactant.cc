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

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>
#include <exadg/structure/material/library/surfactant.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace ExaDG::Structure;

namespace
{
using Model  = WiechertSurfactantModel<3, double>;
using scalar = Model::scalar;
using vector = Model::vector;
using tensor = Model::tensor;

double const time_step_size = 0.1;

WiechertSurfactantData
make_data()
{
  WiechertSurfactantData data;
  data.boundary_ids               = {7, 9};
  data.equilibrium_time           = 1.0;
  data.m_1                        = 48.0;
  data.m_2                        = 20.0;
  data.k_1                        = 2.0;
  data.k_2                        = 1.0;
  data.c                          = 0.5;
  data.relative_concentration_max = 2.0;
  data.gamma_ref                  = 70.0;
  data.gamma_eq                   = 22.0;
  data.gamma_min                  = 2.0;
  return data;
}

Model
make_initialized_model(WiechertSurfactantData const & data, double const surface_area = 1.0)
{
  Model model(data, 1);
  model.update(dealii::make_vectorized_array<double>(surface_area),
               data.equilibrium_time,
               time_step_size,
               0);
  return model;
}

tensor
zero_gradient()
{
  return tensor{};
}

vector
first_material_normal()
{
  vector result{};
  result[0] = 1.0;
  return result;
}

vector
negative_first_material_normal()
{
  vector result{};
  result[0] = -1.0;
  return result;
}

void
check_close(double const value,
            double const reference,
            double const tolerance,
            char const * message)
{
  if(std::abs(value - reference) > tolerance)
  {
    std::ostringstream output;
    output << message << " (value=" << value << ", reference=" << reference
           << ", tolerance=" << tolerance << ")";
    throw std::runtime_error(output.str());
  }
}

void
check_identity_surface_tensor(Model &      model,
                              double const surface_area,
                              double const expected_gamma,
                              double const time,
                              char const * message)
{
  tensor const P = model.surface_tension_1PK(zero_gradient(),
                                             first_material_normal(),
                                             dealii::make_vectorized_array<double>(surface_area),
                                             time,
                                             time_step_size,
                                             0);

  check_close(P[0][0][0], 0.0, 1.0e-12, message);
  check_close(P[1][1][0], expected_gamma, 1.0e-12, message);
  check_close(P[2][2][0], expected_gamma, 1.0e-12, message);
  check_close(P[0][1][0], 0.0, 1.0e-12, message);
  check_close(P[1][0][0], 0.0, 1.0e-12, message);
}

void
check_boundary_ids()
{
  WiechertSurfactantData data = make_data();

  Model model(data, 1);

  Assert(model.is_surfactant_boundary(7),
         dealii::ExcMessage("registered boundary was not detected"));
  Assert(model.is_surfactant_boundary(9),
         dealii::ExcMessage("second registered boundary was not detected"));
  Assert(not model.is_surfactant_boundary(8),
         dealii::ExcMessage("unregistered boundary was detected"));
}

void
check_equilibration()
{
  WiechertSurfactantData data = make_data();
  Model                  model(data, 1);
  model.update(dealii::make_vectorized_array<double>(1.0),
               data.equilibrium_time,
               time_step_size,
               0);

  check_identity_surface_tensor(model, 1.0, 0.0, 0.0, "initial equilibration stress is wrong");

  check_identity_surface_tensor(
    model, 1.0, 11.0, 0.5, "equilibration stress is not linearly ramped");

  check_identity_surface_tensor(
    model, 1.0, 22.0, data.equilibrium_time, "equilibrium stress is wrong");
}

void
check_regimes()
{
  WiechertSurfactantData data = make_data();

  {
    Model model = make_initialized_model(data);
    check_identity_surface_tensor(model, 0.8, 22.0 - 20.0 * 0.25, 1.1, "regime 2 stress is wrong");
  }

  {
    Model model = make_initialized_model(data);
    check_identity_surface_tensor(model, 0.4, 2.0, 1.1, "regime 3 stress is wrong");
  }

  {
    Model model = make_initialized_model(data);
    model.update(dealii::make_vectorized_array<double>(4.0), 1.1, time_step_size, 0);

    double const concentration_new = (0.25 * 4.0 * 10.0 + 4.0 * 1.0) / (4.0 * 12.0);
    double const expected_gamma    = data.gamma_ref - data.m_1 * concentration_new;
    check_identity_surface_tensor(
      model, 4.0, expected_gamma, 1.2, "regime 1 Langmuir update is wrong");
  }
}

void
check_surface_area_derivatives()
{
  WiechertSurfactantData data  = make_data();
  Model                  model = make_initialized_model(data);

  model.update(dealii::make_vectorized_array<double>(4.0), 1.1, time_step_size, 0);

  double const surface_area      = 4.0;
  double const delta_area        = 1.0e-3;
  double const expected_du_gamma = data.m_1 * 0.25 * 4.0 /
                                   (time_step_size * surface_area * surface_area *
                                    (1.0 / time_step_size + data.k_1 * data.c + data.k_2));

  tensor const zero_increment{};
  tensor const tangent = model.surface_tension_1PK_displacement_derivative(
    zero_increment,
    zero_gradient(),
    first_material_normal(),
    dealii::make_vectorized_array<double>(surface_area),
    dealii::make_vectorized_array<double>(delta_area),
    1.2,
    time_step_size,
    0);

  check_close(tangent[1][1][0],
              expected_du_gamma * delta_area,
              1.0e-11,
              "regime 1 surface-area derivative is wrong");

  Model        regime_2_model   = make_initialized_model(data);
  tensor const regime_2_tangent = regime_2_model.surface_tension_1PK_displacement_derivative(
    zero_increment,
    zero_gradient(),
    first_material_normal(),
    dealii::make_vectorized_array<double>(0.8),
    dealii::make_vectorized_array<double>(delta_area),
    1.2,
    time_step_size,
    0);
  double const expected_regime_2_du_gamma = data.m_2 / (0.8 * 0.8);
  check_close(regime_2_tangent[1][1][0],
              expected_regime_2_du_gamma * delta_area,
              1.0e-11,
              "regime 2 surface-area derivative is wrong");

  Model        regime_3_model   = make_initialized_model(data);
  tensor const regime_3_tangent = regime_3_model.surface_tension_1PK_displacement_derivative(
    zero_increment,
    zero_gradient(),
    first_material_normal(),
    dealii::make_vectorized_array<double>(0.4),
    dealii::make_vectorized_array<double>(delta_area),
    1.2,
    time_step_size,
    0);
  check_close(regime_3_tangent[1][1][0],
              0.0,
              1.0e-12,
              "regime 3 surface-area derivative is not zero");
}

void
check_surface_geometry()
{
  WiechertSurfactantData data  = make_data();
  Model                  model = make_initialized_model(data);

  tensor displacement_gradient{};
  for(std::size_t v = 0; v < scalar::size(); ++v)
    displacement_gradient[1][1][v] = 0.2;

  tensor const P = model.surface_tension_1PK(displacement_gradient,
                                             first_material_normal(),
                                             dealii::make_vectorized_array<double>(1.2),
                                             0.5,
                                             time_step_size,
                                             0);
  check_close(P[1][1][0], 11.0, 1.0e-12, "tangential surface geometry is wrong");
  check_close(P[2][2][0], 13.2, 1.0e-12, "surface-area stretch is wrong");

  tensor const P_reversed = model.surface_tension_1PK(displacement_gradient,
                                                      negative_first_material_normal(),
                                                      dealii::make_vectorized_array<double>(1.2),
                                                      0.5,
                                                      time_step_size,
                                                      0);
  for(unsigned int i = 0; i < 3; ++i)
    for(unsigned int j = 0; j < 3; ++j)
      check_close(P_reversed[i][j][0],
                  P[i][j][0],
                  1.0e-12,
                  "surface tensor depends on normal orientation");
}

void
check_full_tangent(double const surface_area, double const time)
{
  WiechertSurfactantData data  = make_data();
  Model                  model = make_initialized_model(data);

  tensor increment{};
  for(std::size_t v = 0; v < scalar::size(); ++v)
    increment[1][1][v] = 0.03;

  scalar const area_increment = dealii::make_vectorized_array<double>(surface_area * 0.03);
  tensor const analytical =
    model.surface_tension_1PK_displacement_derivative(increment,
                                                      zero_gradient(),
                                                      first_material_normal(),
                                                      dealii::make_vectorized_array<double>(
                                                        surface_area),
                                                      area_increment,
                                                      time,
                                                      time_step_size,
                                                      0);

  double const epsilon        = 1.0e-7;
  tensor       gradient_plus  = zero_gradient();
  tensor       gradient_minus = zero_gradient();
  gradient_plus[1][1] += epsilon * increment[1][1];
  gradient_minus[1][1] -= epsilon * increment[1][1];

  tensor const P_plus  = model.surface_tension_1PK(gradient_plus,
                                                  first_material_normal(),
                                                  dealii::make_vectorized_array<double>(
                                                    surface_area * (1.0 + epsilon * 0.03)),
                                                  time,
                                                  time_step_size,
                                                  0);
  tensor const P_minus = model.surface_tension_1PK(gradient_minus,
                                                   first_material_normal(),
                                                   dealii::make_vectorized_array<double>(
                                                     surface_area * (1.0 - epsilon * 0.03)),
                                                   time,
                                                   time_step_size,
                                                   0);

  check_close((P_plus[1][1][0] - P_minus[1][1][0]) / (2.0 * epsilon),
              analytical[1][1][0],
              1.0e-7,
              "full surfactant tangent is wrong");
}

void
check_vectorized_lanes()
{
  WiechertSurfactantData data  = make_data();
  Model                  model = make_initialized_model(data);

  scalar surface_area = dealii::make_vectorized_array<double>(1.0);
  for(std::size_t v = 0; v < scalar::size(); ++v)
    surface_area[v] = (v % 2 == 0) ? 0.8 : 0.4;

  tensor const P = model.surface_tension_1PK(
    zero_gradient(), first_material_normal(), surface_area, 1.1, time_step_size, 0);

  for(std::size_t v = 0; v < scalar::size(); ++v)
  {
    double const expected = (v % 2 == 0) ? 17.0 : 2.0;
    check_close(P[1][1][v], expected, 1.0e-12, "vectorized lane result is wrong");
  }
}
} // namespace

int
main()
{
  try
  {
    check_boundary_ids();
    check_equilibration();
    check_regimes();
    check_surface_area_derivatives();
    check_surface_geometry();
    check_full_tangent(0.8, 1.1);
    check_full_tangent(0.4, 1.1);
    check_vectorized_lanes();

    std::cout << "Surfactant checks: PASS" << std::endl;
    return 0;
  }
  catch(std::exception const & exception)
  {
    std::cerr << "Surfactant checks: FAIL\n" << exception.what() << std::endl;
    return 1;
  }
}
