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
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <exadg/functions_and_boundary_conditions/function_with_normal.h>
#include <exadg/grid/grid.h>
#include <exadg/grid/grid_data.h>
#include <exadg/structure/material/library/alveolar_tissue.h>
#include <exadg/structure/material/library/compressible_neo_hookean.h>
#include <exadg/structure/material/library/st_venant_kirchhoff.h>
#include <exadg/structure/material/library/st_venant_kirchhoff_alveolar_tissue.h>
#include <exadg/structure/postprocessor/postprocessor.h>
#include <exadg/structure/spatial_discretization/operator.h>
#include <exadg/structure/user_interface/application_base.h>
#include <exadg/structure/user_interface/enum_types.h>
#include <exadg/structure/user_interface/parameters.h>
#include <exadg/utilities/create_directories.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace ExaDG
{
namespace Structure
{
namespace
{
// edge length of the box [0, L]^3 (mm) that the mesh is cut out of
constexpr double BOX_LENGTH = 1.0;

// boundary ids: surfaces of the alveolar voids and the six faces of the box [0, L]^3
constexpr dealii::types::boundary_id BOUNDARY_ALVEOLUS  = 1;
constexpr dealii::types::boundary_id BOUNDARY_BOX_LOW   = 2; // x, y, z = 0 -> 2, 3, 4
constexpr dealii::types::boundary_id BOUNDARY_BOX_HIGH  = 5; // x, y, z = L -> 5, 6, 7
constexpr dealii::types::material_id MATERIAL_ID_TISSUE = 1; // GridIn: material id 1 from .msh

// Alveolar air pressure: linear ramp from 0 to the baseline over [0, equilibrium_time] while the
// surfactant relaxes to its equilibrium surface tension, followed by a cyclic (breathing) pressure
// about the baseline.
inline double
alveolar_pressure(double const time,
                  double const equilibrium_time,
                  double const baseline,
                  double const amplitude,
                  double const period)
{
  if(time < equilibrium_time)
    return baseline * (time / equilibrium_time);

  return baseline +
         amplitude * std::sin(2.0 * dealii::numbers::PI * (time - equilibrium_time) / period);
}

template<int dim>
dealii::Tensor<2, dim>
identity_tensor()
{
  return dealii::Tensor<2, dim>(dealii::unit_symmetric_tensor<dim>());
}
} // namespace

// Alveolar pressure acting on the alveolar surfaces. This is a dead load per reference area,
// t_0 = -p N, since the normal vector is the one of the reference configuration and
// pull_back_traction = false.
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

    double const pressure =
      alveolar_pressure(this->get_time(), equilibrium_time, baseline, amplitude, period);

    dealii::Tensor<1, dim> const n = this->get_normal_vector();

    return -n[c] * pressure;
  }

private:
  double equilibrium_time;
  double baseline;
  double amplitude;
  double period;
};

enum class MacroscopicDeformationPath
{
  Linear, // F = F_k + r (F_{k+1} - F_k) on each segment, e.g., simple shear
  Polar   // F = R^r (I + r (U - I)) F_k with the polar decomposition F_{k+1} F_k^{-1} = R U
};

/*
 * Path from the identity to the deformation gradient A, parametrized by r in [0, 1]:
 * A(r) = R^r (I + r (U - I)) with the polar decomposition A = R U. The rotation is ramped by its
 * angle and the stretch linearly, so that a pure rotation stays stress free along the path.
 */
template<int dim>
class PolarPath
{
public:
  PolarPath(dealii::Tensor<2, dim> const & A)
  {
    static_assert(dim == 2 or dim == 3, "Only implemented for dim = 2, 3.");

    AssertThrow(dealii::determinant(A) > 0.0,
                dealii::ExcMessage("The deformation gradient must have det F > 0."));

    // polar decomposition A = R U with U = sqrt(A^T A)
    auto const eigen = dealii::eigenvectors(dealii::Physics::Elasticity::Kinematics::C(A));
    for(auto const & [eigenvalue, eigenvector] : eigen)
      U += std::sqrt(eigenvalue) * dealii::outer_product(eigenvector, eigenvector);
    dealii::Tensor<2, dim> const R = A * dealii::invert(U);

    // rotation angle and axis
    if constexpr(dim == 2)
    {
      angle = std::atan2(R[1][0], R[0][0]);
    }
    else
    {
      angle = std::acos(std::clamp(0.5 * (dealii::trace(R) - 1.0), -1.0, 1.0));
      if(std::sin(angle) > 1.0e-6)
      {
        axis[0] = R[2][1] - R[1][2];
        axis[1] = R[0][2] - R[2][0];
        axis[2] = R[1][0] - R[0][1];
        axis /= 2.0 * std::sin(angle);
      }
      else if(angle > 1.0)
      {
        // rotation by pi: R = 2 a a^T - I
        unsigned int k = 0;
        for(unsigned int d = 1; d < dim; ++d)
          if(R[d][d] > R[k][k])
            k = d;
        for(unsigned int d = 0; d < dim; ++d)
          axis[d] = 0.5 * (R[d][k] + (d == k ? 1.0 : 0.0));
        axis /= axis.norm();
      }
    }
  }

  dealii::Tensor<2, dim>
  value(double const r) const
  {
    dealii::Tensor<2, dim> const I = identity_tensor<dim>();
    return rotation(r * angle) * (I + r * (U - I));
  }

private:
  // rotation by the given angle about the axis of R (Rodrigues' formula in 3D)
  dealii::Tensor<2, dim>
  rotation(double const phi) const
  {
    dealii::Tensor<2, dim> Q;
    if constexpr(dim == 2)
    {
      Q[0][0] = std::cos(phi);
      Q[0][1] = -std::sin(phi);
      Q[1][0] = std::sin(phi);
      Q[1][1] = std::cos(phi);
    }
    else
    {
      dealii::Tensor<2, dim> axis_cross;
      axis_cross[0][1] = -axis[2];
      axis_cross[0][2] = axis[1];
      axis_cross[1][0] = axis[2];
      axis_cross[1][2] = -axis[0];
      axis_cross[2][0] = -axis[1];
      axis_cross[2][1] = axis[0];

      Q = std::cos(phi) * identity_tensor<dim>() + std::sin(phi) * axis_cross +
          (1.0 - std::cos(phi)) * dealii::outer_product(axis, axis);
    }
    return Q;
  }

  dealii::Tensor<2, dim> U;
  double                 angle = 0.0;
  dealii::Tensor<1, dim> axis;
};

/*
 * Macroscopic deformation gradient F(t), piecewise along the waypoints (t_k, F_k): F = F_0 for
 * t <= t_0, F = F_n for t >= t_n, and in between either linear or polar paths between successive
 * waypoints. For ProblemType::QuasiStatic the time is the load factor in [0, 1].
 */
template<int dim>
class MacroscopicDeformation
{
public:
  MacroscopicDeformation(std::vector<double> const &                 times,
                         std::vector<dealii::Tensor<2, dim>> const & gradients,
                         MacroscopicDeformationPath const            path)
    : times(times), gradients(gradients), path(path)
  {
    AssertThrow(times.size() >= 2 and times.size() == gradients.size(),
                dealii::ExcMessage("A deformation path needs at least two waypoints, with one "
                                   "deformation gradient per time."));

    for(unsigned int k = 0; k < times.size(); ++k)
    {
      AssertThrow(dealii::determinant(gradients[k]) > 0.0,
                  dealii::ExcMessage("The deformation gradients of the path must have det F > 0."));
      if(k + 1 < times.size())
      {
        AssertThrow(times[k + 1] > times[k],
                    dealii::ExcMessage("The times of the deformation path must increase."));
        segments.emplace_back(gradients[k + 1] * dealii::invert(gradients[k]));
      }
    }
  }

  dealii::Tensor<2, dim>
  value(double const time) const
  {
    if(time <= times.front())
      return gradients.front();
    if(time >= times.back())
      return gradients.back();

    unsigned int k = 0;
    while(time > times[k + 1])
      ++k;

    double const r = (time - times[k]) / (times[k + 1] - times[k]);

    if(path == MacroscopicDeformationPath::Linear)
      return gradients[k] + r * (gradients[k + 1] - gradients[k]);

    return segments[k].value(r) * gradients[k];
  }

  bool
  is_diagonal() const
  {
    for(auto const & F : gradients)
      for(unsigned int i = 0; i < dim; ++i)
        for(unsigned int j = 0; j < dim; ++j)
          if(i != j and F[i][j] != 0.0)
            return false;
    return true;
  }

  double
  end_time() const
  {
    return times.back();
  }

private:
  std::vector<double> const                 times;
  std::vector<dealii::Tensor<2, dim>> const gradients;
  MacroscopicDeformationPath const          path;
  std::vector<PolarPath<dim>>               segments;
};

// Macroscopic displacement gradient H(t) = F(t) - I of the periodic problem.
template<int dim>
class MacroscopicDisplacementGradient : public dealii::TensorFunction<2, dim>
{
public:
  MacroscopicDisplacementGradient(MacroscopicDeformation<dim> const & deformation)
    : dealii::TensorFunction<2, dim>(0.0), deformation(deformation)
  {
  }

  dealii::Tensor<2, dim>
  value(dealii::Point<dim> const & p) const final
  {
    (void)p;

    return deformation.value(this->get_time()) - identity_tensor<dim>();
  }

private:
  MacroscopicDeformation<dim> const deformation;
};

// Symmetry boundary conditions: normal displacement (F_dd(t) - 1) L of the plane x_d = L.
template<int dim>
class MacroscopicNormalDisplacementDBC : public dealii::Function<dim>
{
public:
  MacroscopicNormalDisplacementDBC(MacroscopicDeformation<dim> const & deformation,
                                   unsigned int const                  direction)
    : dealii::Function<dim>(dim, 0.0), deformation(deformation), direction(direction)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c) const final
  {
    (void)p;

    if(c != direction)
      return 0.0;

    return (deformation.value(this->get_time())[direction][direction] - 1.0) * BOX_LENGTH;
  }

private:
  MacroscopicDeformation<dim> const deformation;
  unsigned int const                direction;
};

/*
 * Postprocessor of the representative volume element (REV) of lung parenchyma:
 *
 * - writes the total displacement u = H * X + w (instead of the periodic fluctuation w),
 *
 * - computes the macroscopic first Piola-Kirchhoff stress of the REV, i.e., the volume average
 *   over the box [0, L]^3 including the alveolar air and the surface tension,
 *
 *     P_macro = 1/V_0 (int_{solid} P dV + int_{Gamma} P_gamma dA) - p (V_air / V_0) I ,
 *
 *   with the reference volumes V_0 = L^3 and V_air = V_0 - V_solid. The air term -p I is the
 *   stress of the alveolar air consistent with the alveolar pressure, which is applied as a dead
 *   load t_0 = -p N in the reference configuration,
 *
 * - writes the pore (air) volume, the alveolar surface area, and the mean surface tension.
 */
template<int dim, typename Number>
class PostProcessorREV : public PostProcessor<dim, Number>
{
private:
  typedef PostProcessor<dim, Number>                     Base;
  typedef typename PostProcessorBase<Number>::VectorType VectorType;

public:
  PostProcessorREV(PostProcessorData<dim> const &                        pp_data,
                   MPI_Comm const &                                      mpi_comm,
                   bool const                                            write_file,
                   std::string const &                                   filename,
                   double const                                          box_volume,
                   std::function<double(double)> const &                 pressure,
                   std::function<dealii::Tensor<2, dim>(double)> const & deformation_gradient)
    : Base(pp_data, mpi_comm),
      mpi_comm(mpi_comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      write_file(write_file),
      filename(filename),
      box_volume(box_volume),
      pressure(pressure),
      deformation_gradient(deformation_gradient),
      pde_operator(nullptr)
  {
  }

  using Base::setup;

  void
  setup(Operator<dim, Number> const & pde_operator_in) final
  {
    pde_operator = &pde_operator_in;

    Base::setup(pde_operator_in);

    if(write_file and dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::ofstream file(filename, std::ios::trunc);
      file << "# Macroscopic quantities of the REV [0, L]^" << dim << ", L^" << dim << " = "
           << box_volume << " (units: mm, g, s; stress in Pa, surface tension in mN/m)."
           << std::endl
           << "# Columns: time, alveolar pressure p, macroscopic deformation gradient F_ij "
           << "(row-major), macroscopic first Piola-Kirchhoff stress P_ij (row-major), reference "
           << "and current solid volume, current air volume, current porosity, reference and "
           << "current alveolar surface area, mean surface tension." << std::endl;
    }
  }

  void
  do_postprocessing(VectorType const &     solution,
                    double const           time,
                    types::time_step const time_step_number) final
  {
    AssertThrow(pde_operator != nullptr, dealii::ExcMessage("Call setup() first."));

    // output of the total displacement
    VectorType displacement_total;
    pde_operator->compute_total_displacement(displacement_total, solution, time);
    Base::do_postprocessing(displacement_total, time, time_step_number);

    // macroscopic stress
    HomogenizationIntegrals<dim> const integrals =
      pde_operator->compute_homogenization_integrals(solution, time, {BOUNDARY_ALVEOLUS});

    double const                 p     = pressure(time);
    double const                 V_air = box_volume - integrals.volume_reference;
    dealii::Tensor<2, dim> const F     = deformation_gradient(time);
    dealii::Tensor<2, dim> const P_macro =
      (integrals.first_piola_kirchhoff_stress + integrals.surface_tension_stress) / box_volume -
      p * V_air / box_volume * identity_tensor<dim>();

    // The deformed periodic cell has the volume det(F) V_0.
    double const volume_current     = dealii::determinant(F) * box_volume;
    double const air_volume_current = volume_current - integrals.volume_current;
    double const porosity_current   = air_volume_current / volume_current;
    double const surface_tension_mean =
      integrals.surfactant_area_current > 0.0 ?
        integrals.surface_tension_area / integrals.surfactant_area_current :
        0.0;

    // values at round-off level are printed as zero to obtain reproducible (test) output
    double const air_volume_fraction =
      std::abs(V_air / box_volume) < 1.0e-12 ? 0.0 : V_air / box_volume;

    pcout << std::endl
          << "Macroscopic quantities of the REV (t = " << std::scientific << std::setprecision(4)
          << time << "):" << std::endl
          << "  Air volume fraction (reference): " << air_volume_fraction << std::endl
          << "  Deformation gradient F:          " << format_tensor(F) << std::endl
          << "  First Piola-Kirchhoff stress P:  " << format_tensor(P_macro) << std::endl;

    if(write_file and dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::ofstream file(filename, std::ios::app);
      file << std::scientific << std::setprecision(10) << time << " " << p;
      for(unsigned int i = 0; i < dim; ++i)
        for(unsigned int j = 0; j < dim; ++j)
          file << " " << F[i][j];
      for(unsigned int i = 0; i < dim; ++i)
        for(unsigned int j = 0; j < dim; ++j)
          file << " " << P_macro[i][j];
      file << " " << integrals.volume_reference << " " << integrals.volume_current << " "
           << air_volume_current << " " << porosity_current << " " << integrals.area_reference
           << " " << integrals.area_current << " " << surface_tension_mean << std::endl;
    }
  }

private:
  // Entries below 1e-12 times the largest entry are printed as zero.
  static std::string
  format_tensor(dealii::Tensor<2, dim> const & tensor)
  {
    double max_entry = 0.0;
    for(unsigned int i = 0; i < dim; ++i)
      for(unsigned int j = 0; j < dim; ++j)
        max_entry = std::max(max_entry, std::abs(tensor[i][j]));

    std::ostringstream stream;
    stream << std::scientific << std::setprecision(6);
    for(unsigned int i = 0; i < dim; ++i)
    {
      stream << (i == 0 ? "[[" : " [");
      for(unsigned int j = 0; j < dim; ++j)
      {
        double const entry = std::abs(tensor[i][j]) > 1.0e-12 * max_entry ? tensor[i][j] : 0.0;
        stream << (j == 0 ? "" : ", ") << std::setw(13) << entry;
      }
      stream << (i + 1 == dim ? "]]" : "]");
    }
    return stream.str();
  }

  MPI_Comm const             mpi_comm;
  dealii::ConditionalOStream pcout;

  bool const                                          write_file;
  std::string const                                   filename;
  double const                                        box_volume;
  std::function<double(double)> const                 pressure;
  std::function<dealii::Tensor<2, dim>(double)> const deformation_gradient;

  Operator<dim, Number> const * pde_operator;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    // defaults: no macroscopic deformation, pin the vertex closest to the box centre
    F_final_entries.assign(dim * dim, 0.0);
    for(unsigned int d = 0; d < dim; ++d)
      F_final_entries[d * dim + d] = 1.0;

    pinned_point_coordinates.assign(dim, 0.5 * BOX_LENGTH);
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      prm.add_parameter(
        "BoundaryConditions",
        boundary_conditions,
        "Periodic: u = (F - I) X + w with periodic fluctuation w and the macroscopic deformation "
        "gradient F prescribed. Symmetry: symmetry planes x, y, z = 0 and normal displacement "
        "(F_dd - 1) L on x, y, z = L, with tangential sliding (diagonal F only).");
      prm.add_parameter("MacroscopicDeformationGradient",
                        F_final_entries,
                        "Final macroscopic deformation gradient F (dim x dim entries, row-major), "
                        "reached at EquilibriumTime and held afterwards. Ignored if "
                        "DeformationPathTimes is given.");
      prm.add_parameter("DeformationPathTimes",
                        path_times,
                        "Times t_0 < t_1 < ... of the waypoints of the macroscopic deformation "
                        "path (load factors for QuasiStatic). Empty: from I at t = 0 to "
                        "MacroscopicDeformationGradient at EquilibriumTime.");
      prm.add_parameter("DeformationPathGradients",
                        path_gradient_entries,
                        "Deformation gradients F_k (dim x dim entries, row-major, separated by "
                        "','), one per waypoint, separated by ';'.");
      prm.add_parameter("MacroscopicDeformationPath",
                        deformation_path,
                        "Path between waypoints F_k and F_{k+1}: Linear, or Polar, "
                        "F = R^r (I + r (U - I)) F_k with F_{k+1} F_k^{-1} = R U (rotations).");
      prm.add_parameter("PinnedPoint",
                        pinned_point_coordinates,
                        "Periodic: the mesh vertex closest to this point is fixed to remove the "
                        "rigid body translations. It must not lie on the box faces.");
      prm.add_parameter("SolidCube",
                        solid_cube,
                        "Replace the alveolar mesh by a solid cube [0, L]^dim (verification).");
      prm.add_parameter("SolidCubeSubdivisions",
                        solid_cube_subdivisions,
                        "SolidCube: number of cells per edge.");
      prm.add_parameter("MaterialType",
                        material_type,
                        "Wall material: StVenantKirchhoff, CompressibleNeoHookean, or "
                        "OgdenAlveolarTissue (Singh et al. 2024).");
      prm.add_parameter(
        "SurfaceTension",
        surface_tension,
        "Surface tension on the alveolar surfaces: None, Constant "
        "(SurfaceTensionValue), or Wiechert (adsorption-limited surfactant). "
        "Requires ProblemType = Unsteady; not available for CompressibleNeoHookean.");
      prm.add_parameter("SurfaceTensionValue",
                        surface_tension_value,
                        "SurfaceTension = Constant: surface tension in mN/m (= dyn/cm = g/s^2), "
                        "ramped from zero over EquilibriumTime.");
      prm.add_parameter("SurfaceTensionConsistentTangent",
                        surface_tension_consistent_tangent,
                        "Include the consistent tangent of the surface tension (otherwise the "
                        "surface tension is a dead load in the Jacobian).");
      prm.add_parameter("ProblemType",
                        problem_type,
                        "QuasiStatic (load factor in [0, 1], output at start and end) or Unsteady "
                        "(time stepping, output at every time step).");
      prm.add_parameter("Inertia",
                        inertia,
                        "Unsteady: if false, the density is negligible (pseudo-static stepping), "
                        "while the surfactant kinetics and the loads evolve in real time.");
      prm.add_parameter("EndTime",
                        end_time,
                        "Unsteady: end time. If <= 0: end of the deformation path, or "
                        "EquilibriumTime + NCycles * CyclicPeriod, whichever is larger.");
      prm.add_parameter("TimeStepSize",
                        time_step_size,
                        "Unsteady: time step size. If <= 0: CyclicPeriod / StepsPerCycle.");
      prm.add_parameter("ReduceTimeStepOnSolverFailure",
                        reduce_time_step_on_solver_failure,
                        "Unsteady: repeat a failed time step with half the time step size.");
      prm.add_parameter("MaxTimeStepReductions",
                        max_time_step_reductions,
                        "Unsteady: maximum number of time step reductions per time step.");
      prm.add_parameter("LinearSolver",
                        linear_solver,
                        "Linear solver of the Newton iterations: CG (symmetric positive definite "
                        "tangent) or FGMRES (e.g., with the consistent surface tension tangent, "
                        "which is indefinite).");
      prm.add_parameter("UpdatePreconditionerEveryNewtonIterations",
                        update_preconditioner_every_newton_iterations,
                        "Rebuild the (AMG) preconditioner every this many Newton iterations; large "
                        "values rebuild it once per time step.");
      prm.add_parameter("OutputInterval", output_interval, "Time interval of the vtu output.");
      prm.add_parameter("CyclicPeriod", cyclic_period, "Cyclic breathing period (s).");
      prm.add_parameter("CyclicBaseline", cyclic_baseline, "Mean alveolar pressure (ramp target).");
      prm.add_parameter("CyclicAmplitude", cyclic_amplitude, "Cyclic alveolar pressure amplitude.");
      prm.add_parameter("NCycles", n_cycles, "Number of breathing cycles.");
      prm.add_parameter("EquilibriumTime",
                        equilibrium_time,
                        "Load ramp duration (s), also the surfactant equilibration time.");
      prm.add_parameter("StepsPerCycle", steps_per_cycle, "Time steps per cycle (sets dt).");
      prm.add_parameter("LoadIncrement",
                        load_increment,
                        "QuasiStatic: load factor increment per load step, in (0, 1].");
    }
    prm.leave_subsection();
  }

private:
  enum class BoundaryConditionType
  {
    Periodic,
    Symmetry
  };

  enum class SurfaceTensionType
  {
    None,
    Constant,
    Wiechert
  };

  BoundaryConditionType boundary_conditions = BoundaryConditionType::Periodic;

  // final macroscopic deformation gradient (row-major), or the waypoints of the deformation path,
  // and the path between waypoints
  std::vector<double>              F_final_entries;
  std::vector<double>              path_times;
  std::vector<std::vector<double>> path_gradient_entries;
  MacroscopicDeformationPath       deformation_path = MacroscopicDeformationPath::Linear;

  // Periodic: point close to the pinned vertex
  std::vector<double> pinned_point_coordinates;

  bool         solid_cube              = false;
  unsigned int solid_cube_subdivisions = 4;

  MaterialType material_type = MaterialType::StVenantKirchhoff;

  SurfaceTensionType surface_tension                    = SurfaceTensionType::None;
  double             surface_tension_value              = 70.0; // mN/m (water)
  bool               surface_tension_consistent_tangent = false;

  ProblemType  problem_type                                  = ProblemType::QuasiStatic;
  bool         inertia                                       = true;
  double       end_time                                      = -1.0;
  double       time_step_size                                = -1.0;
  bool         reduce_time_step_on_solver_failure            = false;
  unsigned int max_time_step_reductions                      = 10;
  double       output_interval                               = 0.1;
  unsigned int update_preconditioner_every_newton_iterations = 1;
  Solver       linear_solver                                 = Solver::CG;

  double       cyclic_period    = 4.0;   // cyclic breathing period (s)
  double       cyclic_baseline  = 300.0; // ramp target and cyclic mean pressure
  double       cyclic_amplitude = 300.0; // cyclic pressure amplitude
  unsigned int n_cycles         = 3;     // number of breathing cycles
  double       equilibrium_time = 1.0;   // load ramp duration (matches surfactant_data)
  unsigned int steps_per_cycle  = 400;   // time steps per breathing cycle (dt = period/steps)
  double       load_increment   = 0.1;   // QuasiStatic: load factor increment per load step

  static dealii::Tensor<2, dim>
  to_tensor(std::vector<double> const & entries)
  {
    AssertThrow(entries.size() == dim * dim,
                dealii::ExcMessage("A deformation gradient needs dim x dim entries."));

    dealii::Tensor<2, dim> F;
    for(unsigned int i = 0; i < dim; ++i)
      for(unsigned int j = 0; j < dim; ++j)
        F[i][j] = entries[i * dim + j];

    return F;
  }

  MacroscopicDeformation<dim>
  get_macroscopic_deformation() const
  {
    if(path_times.empty())
    {
      return MacroscopicDeformation<dim>({0.0, equilibrium_time},
                                         {identity_tensor<dim>(), to_tensor(F_final_entries)},
                                         deformation_path);
    }

    std::vector<dealii::Tensor<2, dim>> gradients;
    for(auto const & entries : path_gradient_entries)
      gradients.push_back(to_tensor(entries));

    return MacroscopicDeformation<dim>(path_times, gradients, deformation_path);
  }

  double
  get_pressure(double const time) const
  {
    return alveolar_pressure(
      time, equilibrium_time, cyclic_baseline, cyclic_amplitude, cyclic_period);
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type            = problem_type;
    this->param.body_force              = false;
    this->param.large_deformation       = true;
    this->param.pull_back_body_force    = false;
    this->param.pull_back_traction      = false;
    this->param.spatial_integration     = false;
    this->param.force_material_residual = false;
    this->param.stable_formulation      = false;
    this->param.cache_level             = 1;
    this->param.check_type              = 0;

    // The surfactant kinetics need a positive time step size, which the quasi-static driver
    // (load factor in [0, 1], time step size 0) does not provide.
    AssertThrow(surface_tension == SurfaceTensionType::None or
                  problem_type == ProblemType::Unsteady,
                dealii::ExcMessage("SurfaceTension requires ProblemType = Unsteady."));

    // PHYSICAL QUANTITIES
    // Inertia = false removes structural inertia (pseudo-static stepping) by using a negligible
    // density; the unsteady implicit-Euler driver still advances real time so the surfactant
    // kinetics and cyclic pressure continue to evolve.
    this->param.density = inertia ? 1.0e-3 : 1.0e-12; // g / mm^3 (water)
    // weak damping is only defined for ProblemType::Unsteady
    this->param.weak_damping_active      = (problem_type == ProblemType::Unsteady) and inertia;
    this->param.weak_damping_coefficient = inertia ? 1.0 : 0.0;

    // TEMPORAL DISCRETIZATION (implicit Euler = generalized-alpha with rho = 0)
    this->param.start_time = 0.0;
    this->param.end_time   = end_time > 0.0 ? end_time :
                                              std::max(get_macroscopic_deformation().end_time(),
                                                     equilibrium_time + n_cycles * cyclic_period);
    this->param.time_step_size =
      time_step_size > 0.0 ? time_step_size : cyclic_period / steps_per_cycle;
    this->param.gen_alpha_type  = GenAlphaType::GenAlpha;
    this->param.spectral_radius = 0.0; // = implicit Euler (generalized-alpha with rho = 0)
    this->param.reduce_time_step_on_solver_failure = reduce_time_step_on_solver_failure;
    this->param.max_time_step_reductions           = max_time_step_reductions;

    // QuasiStatic: the load factor (= time) runs from 0 to 1 in steps of load_increment
    this->param.load_increment = load_increment;

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
    this->param.solver                         = linear_solver;
    this->param.solver_data                    = SolverData(250, 1.e-6, 1.e-6, 30);
    this->param.solver_info_data.interval_time = 0.1;

    // PERCONDITIONER
    this->param.update_preconditioner                         = true;
    this->param.update_preconditioner_every_time_steps        = 1;
    this->param.update_preconditioner_every_newton_iterations =
      update_preconditioner_every_newton_iterations;

    // Pinned vertices (periodic boundary conditions) are not implemented for multigrid.
    this->param.preconditioner = Preconditioner::AMG;
    // this->param.preconditioner = Preconditioner::Multigrid;

    // this->param.multigrid_data.type                         = MultigridType::pMG;
    // this->param.multigrid_data.p_sequence                   = PSequenceType::DecreaseByOne;
    // this->param.multigrid_data.smoother_data.smoother       = MultigridSmoother::Chebyshev;
    // this->param.multigrid_data.smoother_data.preconditioner =
    // PreconditionerSmoother::PointJacobi; this->param.multigrid_data.smoother_data.iterations =
    // 12; this->param.multigrid_data.smoother_data.relaxation_factor = 0.8; // Jacobi, default: 0.8
    // this->param.multigrid_data.smoother_data.smoothing_range   = 20;  // Chebyshev, default: 20
    // this->param.multigrid_data.smoother_data.iterations_eigenvalue_estimation =
    //   20; // Chebyshev, default: 20
    // this->param.multigrid_data.coarse_problem.solver      = MultigridCoarseGridSolver::AMG;
    // this->param.multigrid_data.coarse_problem.solver_data = SolverData(1e3, 1.e-6, 1.e-6, 30);
    // this->param.multigrid_data.coarse_problem.preconditioner =
    //   MultigridCoarseGridPreconditioner::None;

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
  set_boundary_ids(dealii::Triangulation<dim> & tria) const
  {
    // The mesh is the cube [0, L]^3 cut out of a periodic (BCC) lattice of alveoli, where the box
    // planes are both periodicity planes and mirror symmetry planes of the lattice. With periodic
    // or symmetry boundary conditions, alveoli clipped by the box are completed by their periodic
    // or mirror images, so all void surfaces are surfaces of complete alveoli.
    double const tol = 1.e-8;

    for(auto const & cell : tria.active_cell_iterators())
    {
      for(auto const & face : cell->face_iterators())
      {
        if(not face->at_boundary())
          continue;

        dealii::types::boundary_id id = BOUNDARY_ALVEOLUS;
        for(unsigned int d = 0; d < dim; ++d)
        {
          if(std::abs(face->center()[d] - 0.0) < tol)
            id = BOUNDARY_BOX_LOW + d;
          else if(std::abs(face->center()[d] - BOX_LENGTH) < tol)
            id = BOUNDARY_BOX_HIGH + d;
        }

        face->set_all_boundary_ids(id);
      }
    }
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
      if(solid_cube)
      {
        dealii::GridGenerator::subdivided_hyper_cube(tria,
                                                     solid_cube_subdivisions,
                                                     0.0,
                                                     BOX_LENGTH);
        for(auto const & cell : tria.active_cell_iterators())
          cell->set_material_id(MATERIAL_ID_TISSUE);
      }
      else
      {
        GridUtilities::read_external_triangulation<dim>(tria, this->param.grid);
      }

      set_boundary_ids(tria);

      // The box faces x_d = 0 and x_d = L have to be meshed with matching (translated) faces.
      // collect_periodic_faces() throws if a face has no periodic partner.
      if(boundary_conditions == BoundaryConditionType::Periodic)
      {
        for(unsigned int d = 0; d < dim; ++d)
          dealii::GridTools::collect_periodic_faces(
            tria, BOUNDARY_BOX_LOW + d, BOUNDARY_BOX_HIGH + d, d, periodic_face_pairs);

        tria.add_periodicity(periodic_face_pairs);
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

    auto const zero = [&]() { return std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim); };

    // Dirichlet condition on the components selected by the mask. The prescribed displacements
    // start from zero, so the initial acceleration is zero.
    auto const add_dirichlet = [&](dealii::types::boundary_id const               id,
                                   std::shared_ptr<dealii::Function<dim>> const & displacement,
                                   std::vector<bool> const &                      mask)
    {
      this->boundary_descriptor->dirichlet_bc.insert(pair(id, displacement));
      this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(pair(id, zero()));
      this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(id, mask));
    };

    // Alveolar pressure on all alveolar surfaces
    this->boundary_descriptor->neumann_bc.insert(
      pair(BOUNDARY_ALVEOLUS,
           std::make_shared<CyclicHydrostaticPressureNBC<dim>>(
             equilibrium_time, cyclic_baseline, cyclic_amplitude, cyclic_period)));

    MacroscopicDeformation<dim> const deformation = get_macroscopic_deformation();

    if(boundary_conditions == BoundaryConditionType::Periodic)
    {
      // The box faces are periodic (see create_grid()), u = H(t) X + w with periodic w.
      this->boundary_descriptor->macroscopic_displacement_gradient =
        std::make_shared<MacroscopicDisplacementGradient<dim>>(deformation);

      // Periodicity removes the rigid body rotations of w, but not the translations.
      AssertThrow(pinned_point_coordinates.size() == dim,
                  dealii::ExcMessage("PinnedPoint needs dim coordinates."));
      dealii::Point<dim> pinned_point;
      for(unsigned int d = 0; d < dim; ++d)
        pinned_point[d] = pinned_point_coordinates[d];
      this->boundary_descriptor->pinned_vertices.push_back(pinned_point);
    }
    else if(boundary_conditions == BoundaryConditionType::Symmetry)
    {
      // Exact for a diagonal F and a microstructure that is mirror symmetric with respect to the
      // box planes: the box faces stay planar and the material slides tangentially on them.
      AssertThrow(deformation.is_diagonal(),
                  dealii::ExcMessage("BoundaryConditions = Symmetry requires diagonal "
                                     "macroscopic deformation gradients."));

      for(unsigned int d = 0; d < dim; ++d)
      {
        std::vector<bool> mask(dim, false);
        mask[d] = true;

        // symmetry plane x_d = 0: the normal displacement u_d is fixed
        add_dirichlet(BOUNDARY_BOX_LOW + d, zero(), mask);

        // plane x_d = L: normal displacement (F_dd - 1) L, free to slide tangentially
        add_dirichlet(BOUNDARY_BOX_HIGH + d,
                      std::make_shared<MacroscopicNormalDisplacementDBC<dim>>(deformation, d),
                      mask);
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("BoundaryConditions not implemented."));
    }
  }

  /*
   * Surfactant model on the alveolar surfaces. Unit system: grams, seconds, millimeters
   * (force = g*mm/s^2). Surface tension gamma is a force/length quantity in g/s^2, and
   * 1 dyn/cm = 1 mN/m = 1 g/s^2, so the literature dyn/cm numbers are used as-is.
   */
  WiechertSurfactantData
  get_surfactant_data() const
  {
    WiechertSurfactantData data;

    data.boundary_ids       = std::unordered_set<dealii::types::boundary_id>{BOUNDARY_ALVEOLUS};
    data.equilibrium_time   = equilibrium_time;
    data.consistent_tangent = surface_tension_consistent_tangent;

    if(surface_tension == SurfaceTensionType::Constant)
    {
      // Constant surface tension: no adsorption (k_1 = k_2 = 0) and a flat isotherm,
      // gamma_ref = gamma_eq = gamma (m_1 = 0) and a negligible slope m_2 of the second isotherm.
      data.gamma_ref                  = surface_tension_value;
      data.gamma_eq                   = surface_tension_value;
      data.m_1                        = 0.0;
      data.m_2                        = 1.0e-9 * surface_tension_value;
      data.relative_concentration_max = 2.0;
      data.gamma_min = data.gamma_eq - data.m_2 * (data.relative_concentration_max - 1.0);
      data.k_1       = 0.0;
      data.k_2       = 0.0;
      data.c         = 0.0;
    }
    else
    {
      // Otis 1994 / Denny & Schroter 2000 isotherm, Wiechert et al. 2009. For SurfaceTension =
      // None, these values only have to be valid, as the surfactant boundaries are removed below.
      data.gamma_ref = 70.0;                           // dyn/cm (water)
      data.gamma_eq  = 22.0;                           // dyn/cm (Denny & Schroter)
      data.gamma_min = 2.0;                            // dyn/cm (Denny & Schroter)
      data.m_1       = data.gamma_ref - data.gamma_eq; // 48 dyn/cm
      data.m_2       = 81.3;                           // dyn/cm (Otis 1994, graphically)
      data.relative_concentration_max =
        1.0 + (data.gamma_eq - data.gamma_min) / data.m_2; // ~1.246 (second isotherm)
      data.k_1 = 1.6e5;                                    // mm^3 / mg / s (Denny and Schroter)
      data.k_2 = 0.016;                                    // 1 / s         (Denny and Schroter)
      data.c   = 7.3e-6;                                   // mg / mm^3     (Denny and Schroter)
    }

    if(surface_tension == SurfaceTensionType::None)
      data.boundary_ids.clear();

    return data;
  }

  void
  set_material_descriptor() final
  {
    std::shared_ptr<MaterialData> material;

    // E = 2 * G * (1 + nu), for alveolar: G ~= 2.1e3 (see Ogden below)
    double const E_svk  = 5.46e3;
    double const nu_svk = 0.3;

    if(material_type == MaterialType::StVenantKirchhoff)
    {
      if(surface_tension == SurfaceTensionType::None)
      {
        material = std::make_shared<StVenantKirchhoffData<dim>>(MaterialType::StVenantKirchhoff,
                                                                E_svk,
                                                                nu_svk,
                                                                Type2D::Undefined);
      }
      else
      {
        material = std::make_shared<StVenantKirchhoffAlveolarTissueData<dim>>(
          MaterialType::StVenantKirchhoffAlveolarTissue,
          E_svk,
          nu_svk,
          Type2D::Undefined,
          get_surfactant_data());
      }
    }
    else if(material_type == MaterialType::CompressibleNeoHookean)
    {
      AssertThrow(surface_tension == SurfaceTensionType::None,
                  dealii::ExcMessage("CompressibleNeoHookean has no surface tension."));

      // Small-strain moduli of the Ogden wall model below: shear modulus
      // 1/2 (mu1 alpha1 + mu2 alpha2) ~= 2.1e3 and bulk modulus kappa = 2.1e4.
      double const shear_modulus = 0.5 * (6.24 * 16.714 + 935.0 * 4.456);
      double const bulk_modulus  = 2.1e4;

      // The ExaDG energy contains lambda * ln(J)^2 (not lambda / 2 * ln(J)^2), so its lambda is
      // half the first Lame parameter: lambda = (bulk_modulus - 2/3 shear_modulus) / 2 ~= 9.8e3.
      double const lambda = 0.5 * (bulk_modulus - 2.0 / 3.0 * shear_modulus);

      material = std::make_shared<CompressibleNeoHookeanData<dim>>(
        MaterialType::CompressibleNeoHookean, shear_modulus, lambda, Type2D::Undefined);
    }
    else if(material_type == MaterialType::OgdenAlveolarTissue)
    {
      // Singh et al. (2024), alveolar wall, Ogden N = 2, optimized for 30% TLC (Table 2).
      // Stress unit is g/(mm*s^2) = 1 Pa, so 1 kPa = 1000 g/(mm*s^2): the paper's kPa values are
      // multiplied by 1000.
      auto ogden =
        std::make_shared<OgdenAlveolarTissueData<dim>>(MaterialType::OgdenAlveolarTissue);
      ogden->mu1    = 6.24;  // 0.00624 kPa
      ogden->mu2    = 935.0; // 0.935   kPa
      ogden->alpha1 = 16.714;
      ogden->alpha2 = 4.456;
      ogden->kappa  = 2.1e4; // moderate near-incompressibility (~10x shear modulus, nu ~ 0.45)
      ogden->type_two_dim    = Type2D::Undefined;
      ogden->surfactant_data = get_surfactant_data();

      material = ogden;
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage("MaterialType not implemented, use StVenantKirchhoff, "
                                     "CompressibleNeoHookean, or OgdenAlveolarTissue."));
    }

    using Pair = std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>>;
    this->material_descriptor->insert(Pair(MATERIAL_ID_TISSUE, material));
  }

  void
  set_field_functions() final
  {
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));

    // periodic problems: initial fluctuation w = 0, i.e., u = H(t_0) X
    this->field_functions->initial_displacement.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));

    this->field_functions->initial_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = 0.0;
    pp_data.output_data.time_control_data.trigger_interval = output_interval;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree;

    bool const write_file = this->output_parameters.write;
    if(write_file)
      create_directories(this->output_parameters.directory, this->mpi_comm);

    MacroscopicDeformation<dim> const deformation = get_macroscopic_deformation();

    return std::make_shared<PostProcessorREV<dim, Number>>(
      pp_data,
      this->mpi_comm,
      write_file,
      this->output_parameters.directory + this->output_parameters.filename +
        "_macroscopic_quantities.txt",
      std::pow(BOX_LENGTH, dim),
      [this](double const time) { return get_pressure(time); },
      [deformation](double const time) { return deformation.value(time); });
  }
};
} // namespace Structure
} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif
