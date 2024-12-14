#!/bin/bash
#SBATCH --job-name=facemap
#SBATCH --output=error_logs/batch_output_%j.txt
#SBATCH --error=error_logs/batch_error_%j.txt
#SBATCH --partition=branicio-gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00

module purge
module load gcc/11.3.0
module load python/3.10
module load cuda/11.8
module load cudnn/8.4.0.27-11.6

# Path to your environment and directory
ENV_PATH=/project/branicio_73/yazdanim/ML/bin/activate
SCRIPT_DIR=/project/branicio_73/yazdanim/super-marker-tracker/source/
cd $SCRIPT_DIR
mkdir -p error_logs

# Activate virtual environment
source $ENV_PATH

# Start monitoring GPU usage and log it every 10 seconds in the background
nvidia-smi --loop=5 >> error_logs/gpu_usage_$SLURM_JOB_ID.txt &

export CUDA_VISIBLE_DEVICES=0


# Run the script with specific parameters
torchrun --nproc_per_node=1 automain.py >> error_logs/log.txt 2>&1

# torchrun --nproc_per_node=2 29_thpc_convdpad_churhc.py --config ./hp_jsons/hp_run_convdpad.json >> error_logs/log_attention_128org_hp_tune.txt 2>&1
