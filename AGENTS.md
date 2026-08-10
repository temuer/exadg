# Agent Instructions

## Build and Test

- This is a C++17 CMake project built around deal.II 9.7+ with p4est support; CMake fails if `DEAL_II_DIR` is not found or deal.II lacks p4est.
- Configure out of tree from `build/`: `cmake .. -DDEAL_II_DIR=/path/to/deal.II -DPICKUP_TESTS=ON`, then build with `cmake --build . -j2` (CI uses two jobs).
- `scripts/config_exadg.sh` is an alternative configuration helper, but it assumes `$WORKING_DIRECTORY/{dealii,exadg}` and removes CMake cache/build files in the current directory before configuring.
- The root build includes the `exadg` library, all `applications/`, `prototypes/`, `tests/`, and Doxygen; use `cmake --build build --target <target>` for a focused application or target.
- Run the full suite from the build directory with `ctest --output-on-failure`; list available tests with `ctest -N`, and select one or a group with `ctest -R '<regex>'` (for example, `ctest -R 'time_integration/bdf_constants_01'`).
- Tests can invoke MPI. When running as root in a container, export `OMPI_ALLOW_RUN_AS_ROOT=1` and `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`, as CI does.
- Optional integrations are off by default. preCICE builds require `-DEXADG_WITH_PRECICE=ON` and a compatible preCICE installation; FFTW uses `EXADG_WITH_FFTW`, while LIKWID must be enabled with `-DEXADG_WITH_LIKWID=ON` because its CMake option declaration is misspelled.

## Repository Layout

- Reusable library code is under `include/exadg/`, grouped by solver area (`poisson`, `convection_diffusion`, incompressible/compressible flow, structure, FSI, and acoustics); application entrypoints and parameter setups live under `applications/`.
- Application directories generally define an executable with `EXADG_PICKUP_EXE(...)` and may contain a `tests/` directory with reference-output tests; do not assume every application has a test.
- Unit/reference tests are grouped in `tests/time_integration`, `tests/utilities`, and `tests/solvers_and_preconditioners`. Test registration is delegated to deal.II through `EXADG_PICKUP_TESTS()`.
- CMake generates the revision header under the build tree; do not edit generated files or commit build artifacts (`build/` is ignored).

## Style and Verification

- Format C/C++ sources with the repository `.clang-format` (deal.II-derived, Allman braces, two-space indentation, 100-column limit). CI checks `h`, `h.in`, `cc`, and `cpp` files with clang-format 11.1.0 and fails if formatting changes remain.
- Before a change is complete, at minimum rebuild the affected target and run its focused CTest regex; use the full `ctest --output-on-failure` suite when practical.
- Doxygen is configured by `doc/doxygen/Doxyfile` and is built with `cmake --build build --target doxygen`.
