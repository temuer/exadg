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

#ifndef STRUCTURE_MATERIAL_LIBRARY_SURFACTANT
#define STRUCTURE_MATERIAL_LIBRARY_SURFACTANT

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <array>
#include <set>
#include <utility>
#include <vector>

namespace ExaDG
{
namespace Structure
{

struct SurfactantData
{
  std::set<dealii::types::boundary_id> boundary_ids{};
  double                               equilibrium_time{0.0};
  double                               m_1{0.0};
  double                               m_2{0.0};
  double                               k_1{0.0};
  double                               k_2{0.0};
  double                               c{0.0};
  double                               relative_concentration_max{1.0};
  double                               gamma_ref{0.0};
  double                               gamma_eq{0.0};
  double                               gamma_min{0.0};
};

template<int dim, typename Number>
class SurfactantModel
{
public:
  using scalar           = dealii::VectorizedArray<Number>;
  using vector           = dealii::Tensor<1, dim, scalar>;
  using tensor           = dealii::Tensor<2, dim, scalar>;
  using symmetric_tensor = dealii::SymmetricTensor<2, dim, scalar>;

  SurfactantModel(SurfactantData const & data, unsigned int n_boundary_face_batches);

  tensor
  surface_tension_1PK(tensor const &     displacement_gradient,
                      vector const &     material_normal_vector,
                      scalar const &     surface_area_new,
                      double const       time,
                      double const       time_step_size,
                      unsigned int const face) const;

  tensor
  surface_tension_1PK_displacement_derivative(tensor const &     displacement_gradient_increment,
                                              tensor const &     displacement_gradient,
                                              vector const &     material_normal_vector,
                                              scalar const &     surface_area_new,
                                              scalar const &     surface_area_new_increment,
                                              double const       time,
                                              double const       time_step_size,
                                              unsigned int const face) const;

  void
  update(scalar const &     surface_area_new,
         double const       time,
         double const       time_step_size,
         unsigned int const boundary_face_id);

  bool
  is_surfactant_boundary(dealii::types::boundary_id boundary_id) const;

private:
  std::pair<scalar, std::array<unsigned int, scalar::size()>>
  surface_tension_and_current_regime(scalar const &     surface_area_new,
                                     double const       delta_t,
                                     unsigned int const face) const;

  scalar
  surface_tension_displacement_derivative(scalar const & surface_area_new,
                                          scalar const & surface_area_new_increment,
                                          std::array<unsigned int, scalar::size()> const & regime,
                                          double const       time_step_size,
                                          unsigned int const face_id) const;

  SurfactantData const & data;

  struct SurfactantVariables
  {
    scalar relative_concentration;
    scalar surface_area;
  };

  std::vector<SurfactantVariables> variables_old;
};

} // namespace Structure

} // namespace ExaDG

#endif /* STRUCTURE_MATERIAL_LIBRARY_SURFACTANT */