#!/bin/bash

# == Job identity =============================================================
#SBATCH --job-name=rainbow_build_16
#SBATCH --output=logs/rainbow_%j.out
#SBATCH --error=logs/rainbow_%j.err

# == Resource request =========================================================
#SBATCH --partition=ampereq           # GPGPU (Ampere) partition
#SBATCH --nodes=1                    # all 8 GPUs live on one node
#SBATCH --ntasks=1                   # one top-level Python process
#SBATCH --cpus-per-task=16           # cores (multiprocessing spawns 8 workers)
#SBATCH --mem=160G                   # ~160 GB of the 747 GB usable
#SBATCH --gres=gpu:A100-full:2                # 2 A100s

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
module load cuda-uoneasy/12.4.0          
module load gcc-uoneasy/11.3.0

export CUPY_NVCC_GENERATE_FLAGS="--allow-unsupported-compiler"

# Override LD_LIBRARY_PATH to use the module's CUDA, not the system one
export CUDA_HOME=$EBROOTCUDA    # or $EBROOTCUDA — check which exists
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# activate env
source /gpfs01/home/psymf9/miniconda3/bin/activate rainbow_gen

set -u

python -c "
import cupy
cupy.show_config()
"


export CUPY_NVCC_OPT_FLAGS="--allow-unsupported-compiler"
export NVCC_PREPEND_FLAGS="--allow-unsupported-compiler"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# explicitly point to the GCC we want
export CC=$(which gcc)
export CXX=$(which g++)

# Confirm which GCC is being used (should be 11.3.0)
echo "Using GCC from: $(which gcc)"
gcc --version | head -n 1

# Confirm GPU visibility
echo ""
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "------------------"
echo ""

# Confirm nvcc is available (needed for CuPy RawModule backend='nvcc')
echo "nvcc version: $(nvcc --version | tail -1)"
echo ""

# == Set multiprocessing start method explicitly =================================

rm -rf "$SLURM_SUBMIT_DIR/.cupy_cache"
rm -rf ~/.cupy/kernel_cache


export CUPY_CACHE_DIR="$SLURM_SUBMIT_DIR/.cupy_cache"   # keep kernel cache on scratch
export CUPY_CUDA_ARCH_LIST="8.0"	
export CUPY_CUDA_COMPILE_OPTIONS="-arch=sm_80"

export CUDA_VISIBLE_DEVICES=0,1             # expose all 2 A100s

echo "--- System Check ---"
# Check physical driver version
cat /proc/driver/nvidia/version | head -n 1
# Check what nvcc we are actually using in the job
nvcc --version | grep release
echo "--------------------"

python -c "import cupy; print('CuPy is working! Device:', cupy.cuda.Device(0).attributes)"

# == Fix cupy =======================================================================


# trying to fix cupy
python -c 'import cupy; print("cupy version:", cupy.__version__)'

rm -rf ~/.cupy/kernel_cache
rm -rf ~/.cache/cupy
rm -rf "$SLURM_SUBMIT_DIR/.cupy_cache"

# == Run ============================================================================
echo "Launching generate.py ..."
/gpfs01/home/psymf9/miniconda3/envs/rainbow_gen/bin/python /gpfs01/home/psymf9/choice-rf-rt/generation/n-16/generate.py

echo ""
echo "========================================================"
echo "End time : $(date)"
echo "Done."
echo "========================================================"
