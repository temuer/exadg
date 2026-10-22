# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Build, test, formatting, and layout rules live in AGENTS.md and apply here as well:

@AGENTS.md

## Architecture

ExaDG is a matrix-free, high-order discontinuous Galerkin (and continuous FE for structure) PDE framework on top of deal.II. Each physics module under `include/exadg/<module>/` follows the same layered structure:

- `user_interface/`: `parameters.{h,cpp}` (solver parameters), `enum_types.h`, boundary/field/material descriptors, and `application_base.h`, the abstract class an application implements (grid creation, parameters, boundary conditions, field functions, postprocessor). `declare_get_application.h` / `implement_get_application.h` wire an application into the generic `main()`.
- `spatial_discretization/`: the PDE operator (`operator.{h,cpp}`) built from matrix-free kernels in `operators/`.
- `time_integration/`: steady, quasi-static, or time-stepping drivers (BDF, generalized-alpha, ...).
- `preconditioners/`, `postprocessor/`: multigrid preconditioners and output/error evaluation.
- `driver.{h,cpp}`: owns grid, operator, time integrator, and postprocessor; `setup()` then `solve()`.
- `solver.h`: contains `main()`. It reads the JSON input file and loops over degree / space refinement / time refinement (convergence studies) from the `SpatialResolution` and `TemporalResolution` sections, creating a fresh application + driver per run.

Shared infrastructure (grid handling, generic operators/kernels, multigrid, Krylov solvers, time-integration constants, utilities) lives in the other top-level directories of `include/exadg/` and is reused by all modules; coupled modules (FSI, aero-acoustics, flow with transport) compose drivers of the single-field modules.

### Applications

An application directory (e.g. `applications/structure/bar/`) contains:
- `application.h`: defines `Application<dim, Number>` deriving from the module's `ApplicationBase`, declares its own JSON parameters in `add_parameters()`, and includes the module's `implement_get_application.h`.
- `solver.cpp`: just `#include <exadg/<module>/solver.h>` followed by `#include "application.h"`.
- `input.json`: sections `General` (`Precision`, `Dim`, `IsTest`), `SpatialResolution`, `TemporalResolution`, `Application`, `Output`.

Run as `./solver input.json`; `./solver input.json --help` writes a default input file listing all available parameters. `applications/<module>/template/` is the starting point for a new application; new application subdirectories are picked up automatically by the parent `CMakeLists.txt`.

Application tests (`applications/.../tests/`) are pairs `<name>.json` + `<name>.output`, run through deal.II's test machinery against the application's `solver` target. Variants such as `<name>.with_trilinos=true.output` or `<name>.mpirun=2.output` restrict when/how a test runs. Library unit tests in `tests/` are `<name>.cc` + `<name>.output` pairs.

### Library build

`exadg` is a single library whose sources are listed explicitly in `TARGET_SRC` in the root `CMakeLists.txt`; a new `.cpp` file in `include/exadg/` must be added there. `.cpp` files typically end with explicit template instantiations for `dim` 2/3 and `float`/`double` (the `float` variants are used by multigrid).

### Structure materials

Material models live in `include/exadg/structure/material/library/`. Adding one touches: the `MaterialType` enum in `structure/user_interface/enum_types.h`, a data struct for its parameters, the class itself (implementing the `Material` interface in `material.h`), and the `switch` in `material_handler.h` that instantiates it per cell `material_id`. Material tests live in `tests/materials/`.
