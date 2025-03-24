#!/usr/bin/env bash
#SBATCH --job-name=ax_submission              # Job name
#SBATCH --ntasks=1                      # Number of MPI tasks to request
#SBATCH --cpus-per-task=30               # Number of CPU cores per MPI task
#SBATCH --mem=8G                        # Total memory to request
#SBATCH --time=0-48:00:00               # Time limit (DD-HH:MM:SS)
#SBATCH --account=chem-electro-2024        # Project account to use
#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hll537@york.ac.uk   # Where to send mail
#SBATCH --output=slurm_logs/%x-%j.log
#SBATCH --error=slurm_logs/%x-%j.err
# Abort if any command fails

set -e
module load rclone
OUTPUT_DIR="/users/hll537/Ross_proteins/frontier_results" 
SIMULATION_DIR="/users/hll537/Ross_proteins/frontier_results/indvidual_simulations/"
#FIRST_JOB_ID=$(sbatch --array=0-20 --export=ALL,OUTPUT_DIR=$OUTPUT_DIR ax_submission_2.job | awk '{print $4}')
#PROCESS_JOB_ID=$(sbatch --dependency=afterany:$FIRST_JOB_ID processing_script.job | awk '{print $4}')

PROCESS_JOB_ID=$(sbatch --export=ALL,OUTPUT_DIR=$OUTPUT_DIR ax_get_best.job | awk '{print $4}')
FRONTPOINTS=$(python -c "import pickle; f=open('job_results.pkl','rb'); data=pickle.load(f); print(data['size'])")
KEYPOINTS=$(python -c "import pickle; f=open('job_results.pkl','rb'); data=pickle.load(f); print(data['keys'])")
ARRAY_RANGE="1-($FRONTPOINTS*$KEYPOINTS)"
SIMULATION_JOB_ID=$(sbatch --dependency=afterany:$PROCESS_JOB_ID --array=$ARRAY_RANGE --export=ALL,OUTPUT_DIR=$OUTPUT_DIR ax_final_simulation.job | awk '{print $4}')
python final_results_collater.py $SIMULATION_DIR "saved_simulations_"
rclone copy frontier_results gdrive:M4D2_inference_6/
