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

#ifndef INCLUDE_EXADG_STRUCTURE_MATERIAL_LIBRARY_ST_VENANT_KIRCHHOFF_ALVEOLAR_TISSUE_H_
#define INCLUDE_EXADG_STRUCTURE_MATERIAL_LIBRARY_ST_VENANT_KIRCHHOFF_ALVEOLAR_TISSUE_H_

// ExaDG
#include <exadg/structure/material/library/alveolar_tissue.h>
#include <exadg/structure/material/library/st_venant_kirchhoff.h>
#include <exadg/structure/material/library/surfactant.h>

namespace ExaDG
{
namespace Structure
{
template<int dim>
struct StVenantKirchhoffAlveolarTissueData : public StVenantKirchhoffData<dim>
{
  StVenantKirchhoffAlveolarTissueData(MaterialType const &   type,
                                      double const &         E,
                                      double const &         nu,
                                      Type2D const &         type_two_dim,
                                      WiechertSurfactantData surfactant_data)
    : StVenantKirchhoffData<dim>(type, E, nu, type_two_dim),
      surfactant_data(std::move(surfactant_data))
  {
  }

  WiechertSurfactantData surfactant_data;
};

/*
 * St. Venant-Kirchhoff wall material, S = lambda tr(E) I + 2 mu E, with the surface tension of the
 * Wiechert surfactant model on the alveolar surfaces. The bulk material is the one of
 * StVenantKirchhoff (finite strains), the surface tension is evaluated by the nonlinear operator
 * through AlveolarSurfactantInterface.
 */
template<int dim, typename Number>
class StVenantKirchhoffAlveolarTissue : public StVenantKirchhoff<dim, Number>,
                                        public AlveolarSurfactantInterface<dim, Number>
{
public:
  StVenantKirchhoffAlveolarTissue(dealii::MatrixFree<dim, Number> const &          matrix_free,
                                  unsigned int const                               dof_index,
                                  unsigned int const                               quad_index,
                                  StVenantKirchhoffAlveolarTissueData<dim> const & data)
    : StVenantKirchhoff<dim, Number>(matrix_free,
                                     dof_index,
                                     quad_index,
                                     data,
                                     true /* large_deformation */),
      surfactant_model(data.surfactant_data, matrix_free.n_boundary_face_batches())
  {
  }

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
  WiechertSurfactantModel<dim, Number> surfactant_model;
};

} // namespace Structure
} // namespace ExaDG

#endif /* INCLUDE_EXADG_STRUCTURE_MATERIAL_LIBRARY_ST_VENANT_KIRCHHOFF_ALVEOLAR_TISSUE_H_ */
