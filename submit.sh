#!/bin/bash
#SBATCH --job-name=volgan_with_heston
#SBATCH --array=0-4
#SBATCH --time=4:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cjin94@uwo.ca


# Set up output folders
JOB_DIR="output_${SLURM_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_DIR"

module load python/3.11

source /home/cjcomp/projects/def-ankush/Diffusion_CJ/cj_venv/bin/activate

cd /home/cjcomp/projects/def-ankush/volganCode/

# Save the Python and submit script in the job folder for reproducibility
mkdir -p "output_${SLURM_JOB_ID}"
cp bce_check.py "output_${SLURM_JOB_ID}/"
cp VolGAN.py "output_${SLURM_JOB_ID}/"
cp "$0" "output_${SLURM_JOB_ID}/submit.sh"

# Run the script and redirect stdout/stderr to task-specific files
python bce_check.py --run_id $SLURM_ARRAY_TASK_ID > "$JOB_DIR/output.log" 2> "$JOB_DIR/error.log"