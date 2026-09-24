#!/usr/bin/env bash
#
# Frequency sweep for the dynamic alveolus_shell simulation.
#
# For each cyclic period in PERIODS, generates a per-case input JSON (with the
# cyclic parameters read at runtime) and runs the solver with `mpirun -np 8`.
# Each case writes to its own output directory so results do not clobber.
#
# Requires a prior build:
#   cmake --build build --target structure_alveolus_shell -j2
#
# Usage:
#   bash applications/structure/alveolus_shell/run_frequency_sweep.sh
#
set -u

EXADG_DIR="/home/temuer/Desktop/workspace/exadg/"
SOLVER="${EXADG_DIR}/build/applications/structure/alveolus_shell/solver"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

NPROC=8

# Cyclic periods (seconds). tau_surfactant ~ 0.84 s, so this spans
# fast (strong surfactant lag) to slow (near-equilibrium each cycle).
PERIODS=(0.5 1 2 4 8)

BASELINE=300.0
AMPLITUDE=300.0
NCYCLES=3
EQUILIBRIUM_TIME=1.0
STEPS_PER_CYCLE=800

for PERIOD in "${PERIODS[@]}"; do
  CASE_DIR="${SCRIPT_DIR}/output/alveolus_shell_freq/${PERIOD}"
  INPUT="${SCRIPT_DIR}/input_freq_${PERIOD}.json"
  LOG="${CASE_DIR}/run.log"

  mkdir -p "${CASE_DIR}"

  cat > "${INPUT}" <<EOF
{
  "General": {
    "Precision": "double",
    "Dim": "3",
    "IsTest": "false"
  },
  "SpatialResolution": {
    "DegreeMin": "2",
    "DegreeMax": "2",
    "RefineSpaceMin": "0",
    "RefineSpaceMax": "0"
  },
  "TemporalResolution": {
    "RefineTimeMin": "0",
    "RefineTimeMax": "0"
  },
  "Application": {
    "CyclicPeriod": "${PERIOD}",
    "CyclicBaseline": "${BASELINE}",
    "CyclicAmplitude": "${AMPLITUDE}",
    "NCycles": "${NCYCLES}",
    "EquilibriumTime": "${EQUILIBRIUM_TIME}",
    "StepsPerCycle": "${STEPS_PER_CYCLE}"
  },
  "Output": {
    "OutputDirectory": "${CASE_DIR}/",
    "OutputName": "alveolus_shell_f${PERIOD}",
    "WriteOutput": "true"
  }
}
EOF

  echo "=== Running period = ${PERIOD} s (end_time = $(echo "${EQUILIBRIUM_TIME} + ${NCYCLES} * ${PERIOD}" | bc) s, dt = $(echo "scale=6; ${PERIOD} / ${STEPS_PER_CYCLE}" | bc) s) ==="
  mpirun -np "${NPROC}" "${SOLVER}" "${INPUT}" 2>&1 | tee "${LOG}"
done

echo "Sweep complete. Results under ${SCRIPT_DIR}/output/alveolus_shell_freq/"
