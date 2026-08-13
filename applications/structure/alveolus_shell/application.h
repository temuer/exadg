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

#ifndef STRUCTURE_ALVEOLAR
#define STRUCTURE_ALVEOLAR

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <exadg/functions_and_boundary_conditions/function_with_normal.h>
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_data.h>
#include <exadg/structure/material/library/alveolar_tissue.h>
#include <exadg/structure/postprocessor/postprocessor.h>
#include <exadg/structure/user_interface/application_base.h>
#include <exadg/structure/user_interface/enum_types.h>
#include <exadg/structure/user_interface/parameters.h>
#include <memory>
#include <string>

namespace ExaDG
{
namespace Structure
{
template<int dim>
class HydrostaticPressureNBC : public FunctionWithNormal<dim>
{
public:
  HydrostaticPressureNBC(double const pressure, bool const is_quasistatic)
    : FunctionWithNormal<dim>(dim, 0.0), pressure(pressure), is_quasistatic(is_quasistatic)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;

    double factor = is_quasistatic ? this->get_time() : 1.0;

    dealii::Tensor<1, dim> const n = this->get_normal_vector();

    return -n[c] * factor * pressure;
  }

private:
  double pressure{0.0};
  bool   is_quasistatic{false};
};

template<int dim>
class DisplacementDBC : public dealii::Function<dim>
{
public:
  DisplacementDBC(double const displacement, bool const is_quasistatic)
    : dealii::Function<dim>(dim), displacement(displacement), is_quasistatic(is_quasistatic)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;

    double factor = is_quasistatic ? this->get_time() : 1.0;

    if(c == 2)
      return displacement * factor;
    else
      return 0.0;
  }

private:
  double displacement;
  bool   is_quasistatic;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type            = ProblemType::QuasiStatic;
    this->param.body_force              = false;
    this->param.large_deformation       = true;
    this->param.pull_back_body_force    = false;
    this->param.pull_back_traction      = false;
    this->param.spatial_integration     = false;
    this->param.force_material_residual = false;
    this->param.stable_formulation      = false;
    this->param.cache_level             = 0;
    this->param.check_type              = 0;

    // PHYSICAL QUANTITIES
    this->param.density             = 1.0e3;
    this->param.weak_damping_active = false;

    // TEMPORAL DISCRETIZATION
    this->param.load_increment = 0.0125;

    // SPATIAL DISCRETIZATION
    this->param.grid.file_name = this->grid_parameters.file_name;
    // this->param.grid.element_type                 = ElementType::Simplex;
    // this->param.grid.triangulation_type           = TriangulationType::FullyDistributed;
    this->param.grid.element_type                 = ElementType::Hypercube;
    this->param.grid.triangulation_type           = TriangulationType::Serial;
    this->param.grid.create_coarse_triangulations = false;
    this->param.mapping_degree                    = 2;
    this->param.use_matrix_based_implementation   = false;

    // SOLVER
    this->param.newton_solver_data = Newton::SolverData(100, 1.e-4, 1.e-6);
    this->param.solver             = Solver::CG;
    this->param.solver_data        = SolverData(250, 1.e-4, 1.e-6, 30);

    // PERCONDITIONER
    this->param.update_preconditioner                         = true;
    this->param.update_preconditioner_every_time_steps        = 1;
    this->param.update_preconditioner_every_newton_iterations = 1;

    this->param.preconditioner = Preconditioner::AMG;
    // this->param.multigrid_data.type                         = MultigridType::pMG;
    // this->param.multigrid_data.p_sequence                   = PSequenceType::DecreaseByOne;
    // this->param.multigrid_data.smoother_data.smoother       = MultigridSmoother::Chebyshev;
    // this->param.multigrid_data.smoother_data.preconditioner =
    // PreconditionerSmoother::PointJacobi; this->param.multigrid_data.smoother_data.iterations =
    // 12; this->param.multigrid_data.smoother_data.relaxation_factor = 0.8; // Jacobi, default: 0.8
    // this->param.multigrid_data.smoother_data.smoothing_range   = 20;  // Chebyshev, default: 20
    // this->param.multigrid_data.smoother_data.iterations_eigenvalue_estimation =
    //   20; // Chebyshev, default: 20
    // this->param.multigrid_data.coarse_problem.solver      = MultigridCoarseGridSolver::Chebyshev;
    // this->param.multigrid_data.coarse_problem.solver_data = SolverData(1e3, 1.e-6, 1.e-6, 30);
    // this->param.multigrid_data.coarse_problem.preconditioner =
    //   MultigridCoarseGridPreconditioner::PointJacobi;
#ifdef DEAL_II_WITH_TRILINOS
    this->param.multigrid_data.coarse_problem.amg_data.amg_operator_type =
      AMGOperatorType::Elasticity;

    this->param.multigrid_data.coarse_problem.amg_data.ml_data.elliptic = true;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.higher_order_elements =
      this->param.degree > 1;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.n_cycles              = 2;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.w_cycle               = false;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.aggregation_threshold = 1e-4;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.smoother_sweeps       = 2;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.smoother_overlap      = 2;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.smoother_type         = "Chebyshev";
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.coarse_type = "Amesos-UMFPACK";
#endif
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const triangulation_creator =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements)
    {
      static_cast<void>(periodic_face_pairs);

      dealii::Point<dim> const center;
      dealii::GridGenerator::hyper_shell(tria, center, 500.0, 550.0, 192, false);

      for(auto const & cell : tria.active_cell_iterators())
      {
        for(auto const & face : cell->face_iterators())
        {
          if(!face->at_boundary())
          {
            continue;
          }
          if(face->center().norm() < 525.0)
          {
            face->set_all_boundary_ids(1);
          }

          if(face->center().norm() >= 525.0)
          {
            face->set_all_boundary_ids(2);
          }

          if(std::abs(face->center()[0] - -7.105427357601002e-15) < 1.0e-2 &&
             std::abs(face->center()[1] - 532.7608032226563) < 1.0e-2 &&
             std::abs(face->center()[2] - -96.05324554443359) < 1.0e-2)
          { // LEFT BOUNDARY
            face->set_all_boundary_ids(3);
          }
        }
      }

      if(vector_local_refinements.size() > 0)
        refine_local(tria, vector_local_refinements);

      if(global_refinements > 0)
        tria.refine_global(global_refinements);
    };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            triangulation_creator,
                                                            {} /* no local refinements */);

    // mappings
    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
  }

  void
  set_boundary_descriptor() final
  {
    using pair      = std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>;
    using pair_mask = std::pair<dealii::types::boundary_id, dealii::ComponentMask>;

    // Pressure only from the (1 inside, 2 outside)
    this->boundary_descriptor->neumann_bc.insert(
      pair(1, std::make_shared<HydrostaticPressureNBC<dim>>(0.0006, true)));

    // Free boundary (1 inside, 2 outside)
    this->boundary_descriptor->neumann_bc.insert(
      pair(2, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));

    // single element face (clamped)
    this->boundary_descriptor->dirichlet_bc.insert(
      pair(3, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));
    this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(3, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));
    this->boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(3, std::vector<bool>{true, true, true}));
  }

  void
  set_material_descriptor() final
  {
    // {
    //   typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    //   MaterialType const type = MaterialType::StVenantKirchhoff;
    //   // E-Modulus of Steel in unit = [N/microm^2]
    //   double const E = 2.0 * (3.0e-3) * (1.0 + 0.3), nu = 0.3;
    //   Type2D const two_dim_type = Type2D::PlaneStress;

    //   this->material_descriptor->insert(
    //     Pair(0, new StVenantKirchhoffData<dim>(type, E, nu, two_dim_type)));

    //   return;
    // }

    auto material =
      std::make_shared<FibrousAlveolarTissueData<dim>>(MaterialType::FibrousAlveolarTissue);
    // Ground substance
    material->shear_modulus = 2.0e-3; // kg / s2 / microm = 2 kPa (Wiechert)
    // Fiber
    material->fiber_k_1 = 0.0; // 13.5e-3; // kg / s2 / microm = 13.5 kPa (Wiechert)
    material->fiber_k_2 = 0.0; // 76.5;    // 76.5 (Wiechert)
    // Incompressibility
    material->incompressibility_penalty  = 10.0e-3; // kg / s2 / microm != 10 kPa (Wiechert)
    material->incompressibility_exponent = 1.0;     // 1.0;     // 1 (Wiechert)

    // Surfactant
    material->surfactant_data.boundary_ids     = std::set<dealii::types::boundary_id>{1};
    material->surfactant_data.equilibrium_time = 1.0;
    material->surfactant_data.gamma_ref        = 0.0; // 70 dyn / cm (water)
    material->surfactant_data.gamma_eq  = 0.022; // 22.2 dyn / cm = 22 g / s (Denny and Schroter)
    material->surfactant_data.gamma_min = 0.0;   // 2.0 dyn / cm (Denny and Schroter)

    material->surfactant_data.m_1 =
      material->surfactant_data.gamma_ref - material->surfactant_data.gamma_eq;

    material->surfactant_data.m_2 = 0.0; // 81.3... dyn / cm (Otis, graphically)

    material->surfactant_data.relative_concentration_max =
      1.0 + (material->surfactant_data.gamma_eq - material->surfactant_data.gamma_min) /
              material->surfactant_data.m_2; // (second isotherm)

    material->surfactant_data.k_1 = 0.0; // 160 cm3 / mg / s (Denny and Schroter)
    material->surfactant_data.k_2 = 0.0; // 0.015 1 / s (Denny and Schroter)
    material->surfactant_data.c   = 0.0; // 0.0073 mg / ml (Denny and Schroter)

    using Pair = std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>>;
    this->material_descriptor->insert(Pair(0, material));
  }

  void
  set_field_functions() final
  {
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));

    this->field_functions->initial_displacement.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));

    this->field_functions->initial_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active        = true;
    pp_data.output_data.time_control_data.start_time       = 0.0;
    pp_data.output_data.time_control_data.trigger_interval = 0.1;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree;

    return std::make_shared<PostProcessor<dim, Number>>(pp_data, this->mpi_comm);
  }
};
} // namespace Structure
} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif
