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

#ifndef STRUCTURE_ALVEOLAR_HEX
#define STRUCTURE_ALVEOLAR_HEX

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
// If true, remove structural inertia (pseudo-static stepping) while keeping the
// surfactant kinetics and the cyclic pressure in real time. This is done with the
// unsteady (implicit Euler) driver and a negligible density, NOT with
// ProblemType::QuasiStatic: the built-in quasi-static driver ramps the load only
// 0 -> 1 and passes time_step_size = 0, which the surfactant kinetic model
// requires to be positive (it divides by it), so the surfactant cannot evolve
// quasi-statically.
namespace
{
constexpr bool QUASI_STATIC = false;

// boundary ids: alveolar void surfaces and the six faces of the box [0,1]^3
constexpr dealii::types::boundary_id BOUNDARY_ALVEOLUS = 1;
constexpr dealii::types::boundary_id BOUNDARY_BOX_LOW  = 2; // x = 0, y = 0, z = 0 -> 2, 3, 4
constexpr dealii::types::boundary_id BOUNDARY_BOX_HIGH = 5; // x = 1, y = 1, z = 1 -> 5, 6, 7
} // namespace

template<int dim>
class CyclicHydrostaticPressureNBC : public FunctionWithNormal<dim>
{
public:
  CyclicHydrostaticPressureNBC(double const equilibrium_time,
                               double const baseline,
                               double const amplitude,
                               double const period)
    : FunctionWithNormal<dim>(dim, 0.0),
      equilibrium_time(equilibrium_time),
      baseline(baseline),
      amplitude(amplitude),
      period(period)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;

    double const t = this->get_time();

    double pressure = 0.0;

    if(t < equilibrium_time)
    {
      // Phase 1: ramp pressure linearly from 0 to the baseline while the
      // surfactant relaxes to its equilibrium surface tension.
      pressure = baseline * (t / equilibrium_time);
    }
    else
    {
      // Phase 2: cyclic (breathing) pressure load about the baseline.
      pressure = baseline +
                 amplitude * std::sin(2.0 * dealii::numbers::PI * (t - equilibrium_time) / period);
    }

    dealii::Tensor<1, dim> const n = this->get_normal_vector();

    return -n[c] * pressure;
  }

private:
  double equilibrium_time;
  double baseline;
  double amplitude;
  double period;
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

    prm.enter_subsection("Application");
    {
      prm.add_parameter("CyclicPeriod", cyclic_period, "Cyclic breathing period (s).");
      prm.add_parameter("CyclicBaseline", cyclic_baseline, "Mean cyclic pressure (ramp target).");
      prm.add_parameter("CyclicAmplitude", cyclic_amplitude, "Cyclic pressure amplitude.");
      prm.add_parameter("NCycles", n_cycles, "Number of breathing cycles.");
      prm.add_parameter("EquilibriumTime", equilibrium_time, "Surfactant ramp duration (s).");
      prm.add_parameter("StepsPerCycle", steps_per_cycle, "Time steps per cycle (sets dt).");
    }
    prm.leave_subsection();
  }

private:
  double       cyclic_period    = 4.0;   // cyclic breathing period (s)
  double       cyclic_baseline  = 300.0; // ramp target and cyclic mean pressure
  double       cyclic_amplitude = 300.0; // cyclic pressure amplitude
  unsigned int n_cycles         = 3;     // number of breathing cycles
  double       equilibrium_time = 1.0;   // surfactant ramp duration (matches surfactant_data)
  unsigned int steps_per_cycle  = 400;   // time steps per breathing cycle (dt = period/steps)

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type            = ProblemType::Unsteady;
    this->param.body_force              = false;
    this->param.large_deformation       = true;
    this->param.pull_back_body_force    = false;
    this->param.pull_back_traction      = false;
    this->param.spatial_integration     = false;
    this->param.force_material_residual = false;
    this->param.stable_formulation      = false;
    this->param.cache_level             = 1;
    this->param.check_type              = 0;

    // PHYSICAL QUANTITIES
    // QUASI_STATIC = true removes structural inertia (pseudo-static stepping) by using a
    // negligible density; the unsteady implicit-Euler driver still advances real time so the
    // surfactant kinetics and cyclic pressure continue to evolve.
    this->param.density                  = QUASI_STATIC ? 1.0e-12 : 1.0e-3; // g / mm^3 (water)
    this->param.weak_damping_active      = !QUASI_STATIC;
    this->param.weak_damping_coefficient = QUASI_STATIC ? 0.0 : 1.0;

    // TEMPORAL DISCRETIZATION (implicit Euler = generalized-alpha with rho = 0)
    this->param.start_time      = 0.0;
    this->param.end_time        = equilibrium_time + n_cycles * cyclic_period;
    this->param.time_step_size  = cyclic_period / steps_per_cycle;
    this->param.gen_alpha_type  = GenAlphaType::GenAlpha;
    this->param.spectral_radius = 0.0; // = implicit Euler (generalized-alpha with rho = 0)

    // SPATIAL DISCRETIZATION
    this->param.grid.file_name                    = this->grid_parameters.file_name;
    this->param.grid.element_type                 = ElementType::Hypercube;
    this->param.grid.triangulation_type           = TriangulationType::Distributed;
    this->param.grid.create_coarse_triangulations = false;
    this->param.mapping_degree                    = 1;
    this->param.use_matrix_based_implementation   = false;

    // SOLVER
    // With the surfactant treated as a dead load (consistent_tangent = false) the
    // global tangent is symmetric positive definite, so the standard CG solver with
    // a Chebyshev multigrid smoother is appropriate and efficient.
    this->param.newton_solver_data             = Newton::SolverData(100, 1.e-6, 1.e-6);
    this->param.solver                         = Solver::CG;
    this->param.solver_data                    = SolverData(250, 1.e-6, 1.e-6, 30);
    this->param.solver_info_data.interval_time = 0.1;

    // PERCONDITIONER
    this->param.update_preconditioner                         = true;
    this->param.update_preconditioner_every_time_steps        = 1;
    this->param.update_preconditioner_every_newton_iterations = 1;

    // this->param.preconditioner = Preconditioner::AMG;
    this->param.preconditioner = Preconditioner::Multigrid;

    this->param.multigrid_data.type                         = MultigridType::pMG;
    this->param.multigrid_data.p_sequence                   = PSequenceType::DecreaseByOne;
    this->param.multigrid_data.smoother_data.smoother       = MultigridSmoother::Chebyshev;
    this->param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::PointJacobi;
    this->param.multigrid_data.smoother_data.iterations     = 12;
    this->param.multigrid_data.smoother_data.relaxation_factor = 0.8; // Jacobi, default: 0.8
    this->param.multigrid_data.smoother_data.smoothing_range   = 20;  // Chebyshev, default: 20
    this->param.multigrid_data.smoother_data.iterations_eigenvalue_estimation =
      20; // Chebyshev, default: 20
    this->param.multigrid_data.coarse_problem.solver      = MultigridCoarseGridSolver::AMG;
    this->param.multigrid_data.coarse_problem.solver_data = SolverData(1e3, 1.e-6, 1.e-6, 30);
    this->param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::None;

#ifdef DEAL_II_WITH_TRILINOS
    this->param.multigrid_data.coarse_problem.amg_data.amg_operator_type =
      AMGOperatorType::Elasticity;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.elliptic = true;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.higher_order_elements =
      false; // this->param.degree > 1;
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

      GridUtilities::read_external_triangulation<dim>(tria, this->param.grid);

      // The mesh is the 1 x 1 x 1 mm cube [0,1]^3 cut out of a periodic lattice of alveoli. Cells
      // on the box planes are clipped along symmetry planes of the lattice, so the box faces
      // carry symmetry conditions. Every other boundary face belongs to an alveolar void and is
      // loaded by the pressure.
      double const tol = 1.e-8;
      for(auto const & cell : tria.active_cell_iterators())
      {
        for(auto const & face : cell->face_iterators())
        {
          if(!face->at_boundary())
          {
            continue;
          }

          dealii::types::boundary_id id = BOUNDARY_ALVEOLUS;
          for(unsigned int d = 0; d < dim; ++d)
          {
            if(std::abs(face->center()[d] - 0.0) < tol)
              id = BOUNDARY_BOX_LOW + d;
            else if(std::abs(face->center()[d] - 1.0) < tol)
              id = BOUNDARY_BOX_HIGH + d;
          }
          face->set_all_boundary_ids(id);
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

    // Cyclic pressure acting on the alveolar surfaces
    this->boundary_descriptor->neumann_bc.insert(
      pair(BOUNDARY_ALVEOLUS,
           std::make_shared<CyclicHydrostaticPressureNBC<dim>>(
             equilibrium_time, cyclic_baseline, cyclic_amplitude, cyclic_period)));

    // Symmetry planes x = 0, y = 0, z = 0: the normal displacement is fixed, which also removes
    // the rigid body modes.
    for(unsigned int d = 0; d < dim; ++d)
    {
      dealii::types::boundary_id const id = BOUNDARY_BOX_LOW + d;

      this->boundary_descriptor->dirichlet_bc.insert(
        pair(id, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));
      this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
        pair(id, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));

      std::vector<bool> mask(dim, false);
      mask[d] = true;
      this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(id, mask));
    }

    // Planes x = 1, y = 1, z = 1 are free to move outward with the expanding cluster
    for(unsigned int d = 0; d < dim; ++d)
      this->boundary_descriptor->neumann_bc.insert(
        pair(BOUNDARY_BOX_HIGH + d, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));
  }

  void
  set_material_descriptor() final
  {
    auto material =
      std::make_shared<OgdenAlveolarTissueData<dim>>(MaterialType::OgdenAlveolarTissue);

    // Singh et al. (2024), alveolar wall, Ogden N = 2, optimized for 30% TLC (Table 2).
    // Unit system: grams, seconds, millimeters (force = g*mm/s^2).
    // Stress unit is g/(mm*s^2) = 1 Pa, so 1 kPa = 1000 g/(mm*s^2):
    // multiply the paper's kPa values by 1000.
    material->mu1    = 6.24;  // 0.00624 kPa
    material->mu2    = 935.0; // 0.935   kPa
    material->alpha1 = 16.714;
    material->alpha2 = 4.456;
    material->kappa  = 2.1e4; // moderate near-incompressibility (~10x shear modulus, nu ~ 0.45)
    material->type_two_dim = Type2D::Undefined;

    // Surfactant (Otis 1994 / Denny & Schroter 2000 isotherm, Wiechert et al. 2009)
    // Unit system: grams, seconds, millimeters (force = g*mm/s^2).
    // Surface tension gamma is a force/length quantity in g/s^2, and
    // 1 dyn/cm = 1 g/s^2, so the literature dyn/cm numbers are used as-is.
    double const dyn_per_cm_to_app = 1.0;
    material->surfactant_data.boundary_ids =
      std::unordered_set<dealii::types::boundary_id>{BOUNDARY_ALVEOLUS};
    material->surfactant_data.equilibrium_time = equilibrium_time;
    material->surfactant_data.gamma_ref        = 70.0 * dyn_per_cm_to_app; // dyn/cm (water)
    material->surfactant_data.gamma_eq  = 22.0 * dyn_per_cm_to_app; // dyn/cm (Denny & Schroter)
    material->surfactant_data.gamma_min = 2.0 * dyn_per_cm_to_app;  // dyn/cm (Denny & Schroter)

    // Treat the surfactant as a dead load in the Jacobian (omit its consistent
    // follower-load tangent) so the global tangent stays SPD and the multigrid
    // preconditioner remains robust. The surface tension still evolves in the
    // residual, i.e., the surfactant kinetic model keeps responding to area changes.
    material->surfactant_data.consistent_tangent = false;

    material->surfactant_data.m_1 =
      material->surfactant_data.gamma_ref - material->surfactant_data.gamma_eq; // 48 dyn/cm

    material->surfactant_data.m_2 = 81.3 * dyn_per_cm_to_app; // dyn/cm (Otis 1994, graphically)

    material->surfactant_data.relative_concentration_max =
      1.0 + (material->surfactant_data.gamma_eq - material->surfactant_data.gamma_min) /
              material->surfactant_data.m_2; // ~1.246 (second isotherm)

    material->surfactant_data.k_1 = 1.6e5;  // mm^3 / mg / s (Denny and Schroter)
    material->surfactant_data.k_2 = 0.016;  // 1 / s         (Denny and Schroter)
    material->surfactant_data.c   = 7.3e-6; // mg / mm^3     (Denny and Schroter)

    using Pair = std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>>;
    this->material_descriptor->insert(Pair(1, material)); // GridIn: material id 1 from .msh
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
