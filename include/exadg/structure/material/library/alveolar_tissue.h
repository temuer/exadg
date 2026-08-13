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
#include <exadg/structure/material/library/surfactant.h>
#include <exadg/structure/material/material.h>
#include <exadg/structure/user_interface/enum_types.h>

namespace ExaDG
{
namespace Structure
{

template<int dim, typename Number>
class AlveolarSurfactantInterface
{
public:
  virtual ~AlveolarSurfactantInterface() = default;

  virtual WiechertSurfactantModel<dim, Number> const &
  get_surfactant_model() const = 0;

  virtual WiechertSurfactantModel<dim, Number> &
  get_surfactant_model() = 0;
};

template<int dim>
struct FibrousAlveolarTissueData : public MaterialData
{
  FibrousAlveolarTissueData(MaterialType const & type) : MaterialData(type){};

  FibrousAlveolarTissueData(MaterialType const &   type,
                            double const &         shear_modulus,
                            double const &         incompressibility_penalty,
                            double const &         incompressibility_exponent,
                            double const &         fiber_k_1,
                            double const &         fiber_k_2,
                            Type2D const &         type_two_dim,
                            WiechertSurfactantData surfactant_data)
    : MaterialData(type),
      shear_modulus(shear_modulus),
      fiber_k_1(fiber_k_1),
      fiber_k_2(fiber_k_2),
      incompressibility_penalty(incompressibility_penalty),
      incompressibility_exponent(incompressibility_exponent),
      type_two_dim(type_two_dim),
      surfactant_data(std::move(surfactant_data))
  {
  }

  // Ground substance
  double shear_modulus{0.0};

  // Fiber
  double fiber_k_1{0.0};
  double fiber_k_2{0.0};

  // Incompressibility
  double incompressibility_penalty{0.0};
  double incompressibility_exponent{1.0};

  Type2D type_two_dim{Type2D::Undefined};

  WiechertSurfactantData surfactant_data;
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
class FibrousAlveolarTissue : public Material<dim, Number>,
                              public AlveolarSurfactantInterface<dim, Number>
{
public:
  using VectorType     = dealii::LinearAlgebra::distributed::Vector<Number>;
  using Range          = std::pair<unsigned int, unsigned int>;
  using IntegratorCell = CellIntegrator<dim, dim, Number>;

  using scalar           = dealii::VectorizedArray<Number>;
  using vector           = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  using tensor           = dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>;
  using symmetric_tensor = dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>;

  FibrousAlveolarTissue(dealii::MatrixFree<dim, Number> const & matrix_free,
                        unsigned int const                      dof_index,
                        unsigned int const                      quad_index,
                        FibrousAlveolarTissueData<dim> const &  data);

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
                                unsigned int const q) const override;

  symmetric_tensor
  second_piola_kirchhoff_stress_eval(tensor const &     gradient_displacement,
                                     unsigned int const cell,
                                     unsigned int const q) const override;

  symmetric_tensor
  second_piola_kirchhoff_stress(unsigned int const cell, unsigned int const q) const override;

  /*
   * Stress increment Du_S
   */
  symmetric_tensor
  second_piola_kirchhoff_stress_displacement_derivative(tensor const &     gradient_increment,
                                                        tensor const &     gradient_displacement,
                                                        unsigned int const cell,
                                                        unsigned int const q) const override;

  symmetric_tensor
  kirchhoff_stress(tensor const &     gradient_displacement,
                   unsigned int const cell,
                   unsigned int const q) const override;

  symmetric_tensor
  kirchhoff_stress_eval(tensor const &     gradient_displacement,
                        unsigned int const cell,
                        unsigned int const q) const override;

  symmetric_tensor
  kirchhoff_stress(unsigned int const cell, unsigned int const q) const override;

  symmetric_tensor
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          tensor const &           gradient_displacement,
                          unsigned int const       cell,
                          unsigned int const       q) const override;

  symmetric_tensor
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          unsigned int const       cell,
                          unsigned int const       q) const override;

  void
  do_set_cell_linearization_data(
    std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
    unsigned int const                                                         cell) const override;

  scalar
  one_over_J(unsigned int const cell, unsigned int const q) const override;

  tensor
  gradient_displacement(unsigned int const cell, unsigned int const q) const override;

  WiechertSurfactantModel<dim, Number> const &
  get_surfactant_model() const override
  {
    return surfactant_model;
  }

  WiechertSurfactantModel<dim, Number> &
  get_surfactant_model() override
  {
    return surfactant_model;
  }

private:
  unsigned int dof_index;
  unsigned int quad_index;

  FibrousAlveolarTissueData<dim> const & data;

  WiechertSurfactantModel<dim, Number> surfactant_model;
};

template<int dim>
struct NeoHookeAlveolarTissueData : public MaterialData
{
  NeoHookeAlveolarTissueData(MaterialType const & type) : MaterialData(type){};

  NeoHookeAlveolarTissueData(MaterialType const &   type,
                             double const &         E,
                             double const &         nu,
                             Type2D const &         type_two_dim,
                             WiechertSurfactantData surfactant_data)
    : MaterialData(type),
      E(E),
      nu(nu),
      type_two_dim(type_two_dim),
      surfactant_data(std::move(surfactant_data))
  {
  }

  double E{0.0};
  double nu{0.0};

  Type2D type_two_dim{Type2D::Undefined};

  WiechertSurfactantData surfactant_data;
};

/*
 * Psi = E ( 1 - 2 nu) / (4 nu + 4 nu^2) (I_3^( nu / (1 - 2 nu) ) - 1) + E / (4 - 4 nu) * (I_1 - 3)
 */
template<int dim, typename Number>
class NeoHookeAlveolarTissue : public Material<dim, Number>,
                               public AlveolarSurfactantInterface<dim, Number>
{
public:
  using VectorType     = dealii::LinearAlgebra::distributed::Vector<Number>;
  using Range          = std::pair<unsigned int, unsigned int>;
  using IntegratorCell = CellIntegrator<dim, dim, Number>;

  using scalar           = dealii::VectorizedArray<Number>;
  using vector           = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  using tensor           = dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>;
  using symmetric_tensor = dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>;

  NeoHookeAlveolarTissue(dealii::MatrixFree<dim, Number> const & matrix_free,
                         unsigned int const                      dof_index,
                         unsigned int const                      quad_index,
                         NeoHookeAlveolarTissueData<dim> const & data);

  /*
   * S  = E / (2 + 2 nu) ( I - I_3^( - nu / (1 - 2 nu) ) * C^-1 )
   */
  symmetric_tensor
  second_piola_kirchhoff_stress(tensor const &     gradient_displacement,
                                unsigned int const cell,
                                unsigned int const q) const override;

  symmetric_tensor
  second_piola_kirchhoff_stress_eval(tensor const &     gradient_displacement,
                                     unsigned int const cell,
                                     unsigned int const q) const override;

  symmetric_tensor
  second_piola_kirchhoff_stress(unsigned int const cell, unsigned int const q) const override;

  /*
   * Stress increment Du_S
   */
  symmetric_tensor
  second_piola_kirchhoff_stress_displacement_derivative(tensor const &     gradient_increment,
                                                        tensor const &     gradient_displacement,
                                                        unsigned int const cell,
                                                        unsigned int const q) const override;

  symmetric_tensor
  kirchhoff_stress(tensor const &     gradient_displacement,
                   unsigned int const cell,
                   unsigned int const q) const override;

  symmetric_tensor
  kirchhoff_stress_eval(tensor const &     gradient_displacement,
                        unsigned int const cell,
                        unsigned int const q) const override;

  symmetric_tensor
  kirchhoff_stress(unsigned int const cell, unsigned int const q) const override;

  symmetric_tensor
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          tensor const &           gradient_displacement,
                          unsigned int const       cell,
                          unsigned int const       q) const override;

  symmetric_tensor
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          unsigned int const       cell,
                          unsigned int const       q) const override;

  void
  do_set_cell_linearization_data(
    std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
    unsigned int const                                                         cell) const override;

  scalar
  one_over_J(unsigned int const cell, unsigned int const q) const override;

  tensor
  gradient_displacement(unsigned int const cell, unsigned int const q) const override;

  WiechertSurfactantModel<dim, Number> const &
  get_surfactant_model() const override
  {
    return surfactant_model;
  }

  WiechertSurfactantModel<dim, Number> &
  get_surfactant_model() override
  {
    return surfactant_model;
  }

private:
  unsigned int dof_index;
  unsigned int quad_index;

  NeoHookeAlveolarTissueData<dim> const & data;

  WiechertSurfactantModel<dim, Number> surfactant_model;
};


template<int dim>
struct OgdenAlveolarTissueData : public MaterialData
{
  OgdenAlveolarTissueData(MaterialType const & type) : MaterialData(type){};

  OgdenAlveolarTissueData(MaterialType const &   type,
                          double const &         mu1,
                          double const &         mu2,
                          double const &         alpha1,
                          double const &         alpha2,
                          Type2D const &         type_two_dim,
                          WiechertSurfactantData surfactant_data)
    : MaterialData(type),
      mu1(mu1),
      mu2(mu2),
      alpha1(alpha1),
      alpha2(alpha2),
      type_two_dim(type_two_dim),
      surfactant_data(std::move(surfactant_data))
  {
  }

  double mu1{0.0};
  double mu2{0.0};
  double alpha1{0.0};
  double alpha2{0.0};

  Type2D type_two_dim{Type2D::Undefined};

  WiechertSurfactantData surfactant_data;
};

/*
 * Psi = sum_{p=1}^{N} mu_p / alpha_p * (lambda_1^{alpha_p} + lambda_2^{alpha_p} +
 * lambda_3^{alpha_p} - 3)
 */
template<int dim, typename Number>
class OgdenAlveolarTissue : public Material<dim, Number>,
                            public AlveolarSurfactantInterface<dim, Number>
{
public:
  using VectorType     = dealii::LinearAlgebra::distributed::Vector<Number>;
  using Range          = std::pair<unsigned int, unsigned int>;
  using IntegratorCell = CellIntegrator<dim, dim, Number>;

  using scalar           = dealii::VectorizedArray<Number>;
  using vector           = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  using tensor           = dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>;
  using symmetric_tensor = dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>;

  OgdenAlveolarTissue(dealii::MatrixFree<dim, Number> const & matrix_free,
                      unsigned int const                      dof_index,
                      unsigned int const                      quad_index,
                      OgdenAlveolarTissueData<dim> const &    data);

  /*
   * S  = ...
   */
  symmetric_tensor
  second_piola_kirchhoff_stress(tensor const &     gradient_displacement,
                                unsigned int const cell,
                                unsigned int const q) const override;

  symmetric_tensor
  second_piola_kirchhoff_stress_eval(tensor const &     gradient_displacement,
                                     unsigned int const cell,
                                     unsigned int const q) const override;

  symmetric_tensor
  second_piola_kirchhoff_stress(unsigned int const cell, unsigned int const q) const override;

  /*
   * Stress increment Du_S
   */
  symmetric_tensor
  second_piola_kirchhoff_stress_displacement_derivative(tensor const &     gradient_increment,
                                                        tensor const &     gradient_displacement,
                                                        unsigned int const cell,
                                                        unsigned int const q) const override;

  symmetric_tensor
  kirchhoff_stress(tensor const &     gradient_displacement,
                   unsigned int const cell,
                   unsigned int const q) const override;

  symmetric_tensor
  kirchhoff_stress_eval(tensor const &     gradient_displacement,
                        unsigned int const cell,
                        unsigned int const q) const override;

  symmetric_tensor
  kirchhoff_stress(unsigned int const cell, unsigned int const q) const override;

  symmetric_tensor
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          tensor const &           gradient_displacement,
                          unsigned int const       cell,
                          unsigned int const       q) const override;

  symmetric_tensor
  contract_with_J_times_C(symmetric_tensor const & symmetric_gradient_increment,
                          unsigned int const       cell,
                          unsigned int const       q) const override;

  void
  do_set_cell_linearization_data(
    std::shared_ptr<CellIntegrator<dim, dim /* n_components */, Number>> const integrator_lin,
    unsigned int const                                                         cell) const override;

  scalar
  one_over_J(unsigned int const cell, unsigned int const q) const override;

  tensor
  gradient_displacement(unsigned int const cell, unsigned int const q) const override;

  WiechertSurfactantModel<dim, Number> const &
  get_surfactant_model() const override
  {
    return surfactant_model;
  }

  WiechertSurfactantModel<dim, Number> &
  get_surfactant_model() override
  {
    return surfactant_model;
  }

private:
  unsigned int dof_index;
  unsigned int quad_index;

  OgdenAlveolarTissueData<dim> const & data;

  WiechertSurfactantModel<dim, Number> surfactant_model;
};

} // namespace Structure
} // namespace ExaDG

#endif /* STRUCTURE_MATERIAL_LIBRARY_ALVEOLAR_TISSUE */
