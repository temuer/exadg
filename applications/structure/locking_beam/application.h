/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
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

#ifndef STRUCTURE_BEAM
#define STRUCTURE_BEAM

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include "exadg/grid/grid_data.h"
#include "exadg/structure/material/library/st_venant_kirchhoff.h"
#include "exadg/structure/user_interface/enum_types.h"
namespace ExaDG
{
namespace Structure
{

template<int dim>
class AreaLoad : public dealii::Function<dim>
{
public:
  AreaLoad(double force_per_unit_area, bool incremental_loading)
    : dealii::Function<dim>(dim),
      force_per_unit_area(force_per_unit_area),
      incremental_loading(incremental_loading)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;

    double factor = 1.0;
    if(incremental_loading)
      factor = this->get_time();

    if(c == 1)
    {
      return -factor * force_per_unit_area;
    }

    return 0.0;
  }

private:
  double const force_per_unit_area;
  bool const   incremental_loading;
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
      prm.add_parameter("Length", length, "Length of domain.");
      prm.add_parameter("Height", height, "Height of domain.");
      prm.add_parameter("Width", width, "Width of domain.");
      prm.add_parameter("PoissonNumber", poisson_number, "Width of domain.");
      prm.add_parameter("ForcePerUnitArea",
                        force_per_unit_area,
                        "Value of force per unit area on right boundary.");
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
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

    this->param.density             = 1.0e3;
    this->param.weak_damping_active = false;

    // this->param.grid.element_type = ElementType::Hypercube;
    this->param.grid.element_type = ElementType::Simplex;

    this->param.grid.triangulation_type     = TriangulationType::FullyDistributed;
    this->param.mapping_degree              = 1;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;

    this->param.load_increment = 0.01;

    this->param.newton_solver_data = Newton::SolverData(1e3, 1.e-6, 1.e-6);
    this->param.solver             = Solver::CG;
    this->param.solver_data        = SolverData(1e4, 1.e-6, 1.e-6, 100);

    this->param.update_preconditioner                  = true;
    this->param.update_preconditioner_every_time_steps = 1;
    this->param.update_preconditioner_every_newton_iterations =
      this->param.newton_solver_data.max_iter;

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
    this->param.multigrid_data.coarse_problem.solver_data = SolverData(1e4, 1.e-6, 1.e-6, 30);
    this->param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::None;

#ifdef DEAL_II_WITH_TRILINOS
    this->param.multigrid_data.coarse_problem.amg_data.amg_operator_type =
      AMGOperatorType::Elasticity;

    this->param.multigrid_data.coarse_problem.amg_data.ml_data.elliptic = true;
    // this->param.multigrid_data.coarse_problem.amg_data.ml_data.higher_order_elements =
    //   this->param.degree > 1;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.higher_order_elements = false;

    this->param.multigrid_data.coarse_problem.amg_data.ml_data.n_cycles              = 2;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.w_cycle               = false;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.aggregation_threshold = 1e-4;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.smoother_sweeps       = 2;
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.smoother_overlap      = 2;

    this->param.multigrid_data.coarse_problem.amg_data.ml_data.smoother_type = "Chebyshev";
    this->param.multigrid_data.coarse_problem.amg_data.ml_data.coarse_type   = "Amesos-UMFPACK";
#endif
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    (void)mapping;
    (void)multigrid_mappings;

    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements)
    {
      (void)periodic_face_pairs;
      (void)vector_local_refinements;

      dealii::Point<dim> p1, p2;
      p1[0] = 0;
      p1[1] = -(this->height / 2);
      if(dim == 3)
        p1[2] = -(this->width / 2);

      p2[0] = this->length;
      p2[1] = +(this->height / 2);
      if(dim == 3)
        p2[2] = (this->width / 2);

      std::vector<unsigned int> repetitions(dim);
      repetitions[0] = 16;
      repetitions[1] = 2;
      if(dim == 3)
        repetitions[2] = 2;


      // dealii::GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);
      dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices(tria, repetitions, p1, p2);

      constexpr double tol = 1.e-8;

      for(auto cell : tria)
      {
        for(auto const & face : cell.face_indices())
        {
          // left face
          if(std::fabs(cell.face(face)->center()(0) - 0) < tol)
          {
            cell.face(face)->set_all_boundary_ids(1);
          }
          // right face
          else if(std::fabs(cell.face(face)->center()(0) - this->length) < tol)
          {
            cell.face(face)->set_all_boundary_ids(2);
          }
        }
      }

      tria.refine_global(global_refinements);
    };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
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
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    this->boundary_descriptor->neumann_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    // left side
    this->boundary_descriptor->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));

    this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(dim)));

    this->boundary_descriptor->dirichlet_bc_component_mask.insert(
      pair_mask(1, dealii::ComponentMask(std::vector<bool>(dim, true))));

    // right side
    bool const incremental_loading = (this->param.problem_type == ProblemType::QuasiStatic);
    this->boundary_descriptor->neumann_bc.insert(
      pair(2, new AreaLoad<dim>(force_per_unit_area, incremental_loading)));
  }

  void
  set_material_descriptor() final
  {
    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    MaterialType const type = MaterialType::StVenantKirchhoff;

    constexpr double E = 1.0e6;

    Type2D const two_dim_type = Type2D::Undefined;

    std::cout << "----- Poisson Number: " << poisson_number << '\n';

    this->material_descriptor->insert(
      Pair(0, new StVenantKirchhoffData<dim>(type, E, poisson_number, two_dim_type)));
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
    pp_data.output_data.time_control_data.is_active = this->output_parameters.write;
    pp_data.output_data.directory                   = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                    = this->output_parameters.filename;
    pp_data.output_data.write_higher_order          = false;
    pp_data.output_data.degree                      = this->param.degree;

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return post;
  }

  // size of geometry
  double length              = 0.0;
  double height              = 0.0;
  double width               = 0.0;
  double force_per_unit_area = 0.0;
  double poisson_number      = 0.0;
};

} // namespace Structure

} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif
