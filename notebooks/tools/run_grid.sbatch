#!/bin/bash
# =============================================================================
# Title:   Measure the w2_00 chunking grid against the AISC embedding API.
# Purpose: Reads data/IT_Grundschutz_Kompendium_Edition2023.md and the two gold
#          question sets, chunks the corpus 66 ways, embeds every chunking with
#          each of the three models and writes data/grid/*.parquet.
#          The work is API-bound, not compute-bound: no GPU is requested, and the
#          walltime is set by how fast api.aisc.hpi.de answers. About 10 h for a
#          cold run of all 198 configurations, minutes when the cache is warm.
#          AISC hardware is shared and this job may be preempted; --resume makes
#          that harmless, since the tables survive on disk.
# Cluster: written for the HPI sc cluster (SLURM). Every line marked
#          "# ADJUST:" must be changed for a different SLURM cluster.
# Submit:  on a cluster login node ->  sbatch tools/run_grid.sbatch
#          from the workstation     ->  ssh to the login node first (see cluster.env).
#          Run it from the `notebooks/` directory: tools.run_grid is a module.
# Resume:  the tables are rewritten after every configuration, so a job that
#          runs out of walltime is picked up by simply submitting this again.
# =============================================================================

#SBATCH --job-name=rag-chunking-grid          # descriptive kebab-case name; drives %x in logs
#SBATCH --account=aisc                        # ADJUST: cluster account
#SBATCH --qos=aisc                            # ADJUST: QOS for the account
# cpu-batch would suit an API-bound job better, but it only allows the `normal` QOS and this
# account's `normal` association has a submit limit of zero. aisc-batch it is, without a GPU.
#SBATCH --partition=aisc-batch                # ADJUST: batch partition (only aisc-batch accepts sbatch here)
#SBATCH --constraint=ARCH:X86                 # ADJUST: node architecture feature (HPI nodes are x86)
#SBATCH --time=24:00:00                       # ADJUST: walltime <= partition ceiling (aisc-batch: 7 days)
#SBATCH --nodes=1                             # single node
#SBATCH --cpus-per-task=8                     # ADJUST: CPUs per task; the job waits on HTTP, not on cores
#SBATCH --mem=48G                             # ADJUST: RAM; the score matrix is 166 questions x ~50k chunks
#SBATCH --output=logs/%x_%j.out               # %x=job-name, %j=job-id
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# --- environment -------------------------------------------------------------
# No CUDA module: every embedding is an HTTPS call to the workshop API, so the
# node only needs Python and outbound network access.
#
# ADJUST: /sc/home is mounted noexec on this cluster, so neither uv nor the
# virtualenv can live next to the code. Both go on scratch, which is executable.
SCRATCH=/sc/scratch/$USER                     # ADJUST: exec-capable scratch path
export PATH="$SCRATCH/bin:$PATH"
export UV_CACHE_DIR="$SCRATCH/uv-cache"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv-workshop-rag"
command -v uv >/dev/null || { echo "uv is not on PATH; install it into $SCRATCH/bin first" >&2; exit 1; }

# OPENAI_API_KEY and OPENAI_API_BASE come from notebooks/.env, which is
# gitignored and has to be copied to the cluster by hand.
[ -f .env ] || { echo 'notebooks/.env is missing; copy it across before submitting' >&2; exit 1; }

echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-?}  Started: $(date --iso-8601=seconds)"

# --- payload -----------------------------------------------------------------
uv run python -m tools.run_grid --stage dense --resume

echo "Finished: $(date --iso-8601=seconds)"
