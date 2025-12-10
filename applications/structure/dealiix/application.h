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

#ifndef STRUCTURE_DEALIIX
#define STRUCTURE_DEALIIX

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_data.h>
#include <exadg/structure/material/library/st_venant_kirchhoff.h>
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
class ForceNBC : public dealii::Function<dim>
{
public:
  ForceNBC(double const force, bool const is_quasistatic)
    : dealii::Function<dim>(dim, 0.0), force(force), is_quasistatic(is_quasistatic)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    static_cast<void>(p);

    double const factor = is_quasistatic ? this->get_time() : 1.0;

    return c == 0 ? factor * force : 0.0;
  }

private:
  double force{0.0}; // per unit area
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
    static_cast<void>(p);

    double factor = is_quasistatic ? this->get_time() : 1.0;

    return c == 0 ? factor * displacement : 0.0;
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

    // PHYSICAL QUANTITIES
    this->param.density             = 8.0e-6;
    this->param.weak_damping_active = false;

    // TEMPORAL DISCRETIZATION
    this->param.load_increment = 0.1;

    // SPATIAL DISCRETIZATION
    this->param.grid.file_name                    = this->grid_parameters.file_name;
    this->param.grid.element_type                 = ElementType::Hypercube;
    this->param.grid.triangulation_type           = TriangulationType::Serial;
    this->param.grid.create_coarse_triangulations = false;
    this->param.mapping_degree                    = 1;
    this->param.use_matrix_based_implementation   = false;

    // SOLVER
    this->param.newton_solver_data = Newton::SolverData(20, 1.e-12, 1.e-8);
    this->param.solver             = Solver::CG;
    this->param.solver_data        = SolverData(30, 1.e-12, 1.e-8, 30);

    // PERCONDITIONER
    this->param.preconditioner                         = Preconditioner::Multigrid;
    this->param.update_preconditioner                  = true;
    this->param.update_preconditioner_every_time_steps = 1;
    this->param.update_preconditioner_every_newton_iterations =
      this->param.newton_solver_data.max_iter;
    this->param.update_preconditioner_once_newton_converged = true;
    this->param.multigrid_data.type                         = MultigridType::hpMG;
    this->param.multigrid_data.coarse_problem.solver        = MultigridCoarseGridSolver::CG;
    this->param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
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

      dealii::GridGenerator::subdivided_hyper_cube(tria, 8, 0.0, 1.0);

      if(vector_local_refinements.size() > 0)
        refine_local(tria, vector_local_refinements);

      if(global_refinements > 0)
        tria.refine_global(global_refinements);

      for(auto const & cell : tria.active_cell_iterators())
      {
        for(auto const & face : cell->face_iterators())
        {
          if(!face->at_boundary())
          {
            continue;
          }
          if(std::abs(face->center()[0] - 0.0) < 1.0e-8)
          { // LEFT
            face->set_all_boundary_ids(1);
          }
          else if(std::abs(face->center()[0] - 1.0) < 1.0e-8)
          { // RIGHT
            face->set_all_boundary_ids(2);
          }
        }
      }
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
    using Pair     = std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>;
    using PairMask = std::pair<dealii::types::boundary_id, dealii::ComponentMask>;

    // LEFT BOUNDARY (CLAMPED)
    this->boundary_descriptor->dirichlet_bc.insert(
      Pair(1, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));
    this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      Pair(1, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));
    this->boundary_descriptor->dirichlet_bc_component_mask.insert(
      PairMask(1, std::vector<bool>(dim, true)));

    // RIGHT BOUNDARY (DBC)
    this->boundary_descriptor->dirichlet_bc.insert(
      Pair(2, std::make_shared<DisplacementDBC<dim>>(0.1, true)));
    this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      Pair(2, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));
    this->boundary_descriptor->dirichlet_bc_component_mask.insert(
      PairMask(2, std::vector<bool>(dim, true)));

    // RIGHT BOUNDARY (NBC)
    // this->boundary_descriptor->neumann_bc.insert(
    //   Pair(2, std::make_shared<ForceNBC<dim>>(2.0e4, true)));

    // free boundary (boundaries)
    this->boundary_descriptor->neumann_bc.insert(
      Pair(0, std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim)));
  }

  void
  set_material_descriptor() final
  {
    using Pair = std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>>;

    double const E  = 2.0e5;
    double const nu = 0.3;

    this->material_descriptor->insert(Pair(
      0,
      new StVenantKirchhoffData<dim>(MaterialType::StVenantKirchhoff, E, nu, Type2D::Undefined)));
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
    pp_data.output_data.time_control_data.trigger_interval = 1.0 / 2.0;
    pp_data.output_data.directory                          = this->output_parameters.directory;
    pp_data.output_data.filename                           = this->output_parameters.filename;
    pp_data.output_data.write_higher_order                 = true;
    pp_data.output_data.degree                             = this->param.degree;

    return std::make_shared<PostProcessor<dim, Number>>(pp_data, this->mpi_comm);
  }
};

} // namespace Structure
} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif
