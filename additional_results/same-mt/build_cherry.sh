#!/bin/bash

# == Job identity =============================================================
#SBATCH --job-name=cherry_table
#SBATCH --output=logs/rainbow_%j.out
#SBATCH --error=logs/rainbow_%j.err

# == Resource request =========================================================
#SBATCH --partition=hmemq           # GPGPU (Ampere) partition
#SBATCH --nodes=1                    # all 8 GPUs live on one node
#SBATCH --ntasks=1                   # one top-level Python process
#SBATCH --cpus-per-task=16           # cores (multiprocessing spawns 8 workers)
#SBATCH --mem=160G                   # ~160 GB of the 747 GB usable

# == Wall time ================================================================
#SBATCH --time=02:00:00

# == Notifications ============================================================
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=Mahfuzah.Fariha@nottingham.ac.uk

# =============================================================================

set -euo pipefail
set +u

echo "========================================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Start time  : $(date)"
echo "Working dir : $SLURM_SUBMIT_DIR"
echo "========================================================"

# == Move to the directory from which sbatch was called ========================
cd "$SLURM_SUBMIT_DIR"

# == Create log directory if it doesn't exist ==================================
mkdir -p logs

# == Environment setup ==========================================================
module --force purge         
module load gcc-uoneasy/11.3.0

# export CUPY_NVCC_GENERATE_FLAGS="--allow-unsupported-compiler"

# # Override LD_LIBRARY_PATH to use the module's CUDA, not the system one
# export CUDA_HOME=$EBROOTCUDA    
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# activate env
source /gpfs01/home/psymf9/miniconda3/bin/activate rainbow_gen

set -u


# export CUPY_NVCC_OPT_FLAGS="--allow-unsupported-compiler"
# export NVCC_PREPEND_FLAGS="--allow-unsupported-compiler"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# explicitly point to the GCC we want
export CC=$(which gcc)
export CXX=$(which g++)

# Confirm which GCC is being used (should be 11.3.0)
echo "Using GCC from: $(which gcc)"
gcc --version | head -n 1

# == Set multiprocessing start method explicitly =================================

# rm -rf "$SLURM_SUBMIT_DIR/.cupy_cache"
# rm -rf ~/.cupy/kernel_cache


# export CUPY_CACHE_DIR="$SLURM_SUBMIT_DIR/.cupy_cache"   # keep kernel cache on scratch
# export CUPY_CUDA_ARCH_LIST="8.0"	
# export CUPY_CUDA_COMPILE_OPTIONS="-arch=sm_80"

# export CUDA_VISIBLE_DEVICES=0,1,2,3             # expose all 2 A100s

# echo "--- System Check ---"
# # Check physical driver version
# cat /proc/driver/nvidia/version | head -n 1
# # Check what nvcc we are actually using in the job
# nvcc --version | grep release
# echo "--------------------"

# use OpenSSL bundled with conda?
export OPENSSL_PREFIX="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# == Scratch space for compiled .so and any temp files ========================
export TMPDIR="$SLURM_SUBMIT_DIR/.tmp_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"

echo ""
echo "Python       : $(which python)"
python --version

echo ""
echo "OpenSSL      : $(openssl version 2>/dev/null || echo 'cli not found, using conda')"
echo "OPENSSL_PREFIX: $OPENSSL_PREFIX"

# Verify the header and library the compile step will use
echo ""
echo "--- OpenSSL check ---"
ls "$OPENSSL_PREFIX/include/openssl/evp.h"   && echo "evp.h        : found" \
                                              || echo "evp.h        : MISSING â€” compile will fail"
ls "$OPENSSL_PREFIX/lib/libcrypto.so"         && echo "libcrypto.so : found" \
                                              || ls "$OPENSSL_PREFIX/lib64/libcrypto.so" \
                                              && echo "libcrypto.so : found (lib64)" \
                                              || echo "libcrypto.so : MISSING â€” runtime will fail"
echo "---------------------"

# Verify the header and library the compile step will use
echo ""
echo "--- OpenSSL check ---"
ls "$OPENSSL_PREFIX/include/openssl/evp.h"   && echo "evp.h        : found" \
                                              || echo "evp.h        : MISSING â€” compile will fail"
ls "$OPENSSL_PREFIX/lib/libcrypto.so"         && echo "libcrypto.so : found" \
                                              || ls "$OPENSSL_PREFIX/lib64/libcrypto.so" \
                                              && echo "libcrypto.so : found (lib64)" \
                                              || echo "libcrypto.so : MISSING â€” runtime will fail"
echo "---------------------"

# Sanity-check that Python can import everything needed before the full job
echo ""
echo "--- Python import check ---"
python -c "
import numpy, multiprocessing, ctypes, subprocess
from multiprocessing import shared_memory
print('numpy          :', numpy.__version__)
print('multiprocessing: ok')
print('shared_memory  : ok')
print('ctypes         : ok')
print('cpu count      :', multiprocessing.cpu_count())
"
echo "---------------------------"
echo ""

# == Clean up any stale shared memory segments from previous failed runs ======
# If a previous job crashed without reaching the finally block, orphaned shm
# segments may still exist. This clears any named by this user.
python -c "
import os, glob
# Linux exposes POSIX shm as files under /dev/shm
for f in glob.glob('/dev/shm/*'):
    try:
        if os.stat(f).st_uid == os.getuid():
            os.unlink(f)
            print(f'Removed stale shm: {f}')
    except Exception:
        pass
" 2>/dev/null || true

# == Run ============================================================================
echo "Launching generate.py ..."
/gpfs01/home/psymf9/miniconda3/envs/rainbow_gen/bin/python /gpfs01/home/psymf9/choice-rf-rt/add_results/same_mt/cherry/0.95/cherry.py
echo ""
echo "========================================================"
echo "End time : $(date)"
echo "Done."
echo "========================================================"

# == Cleanup ==================================================================
rm -rf "$TMPDIR"