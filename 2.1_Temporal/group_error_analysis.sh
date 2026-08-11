#!/bin/bash
#SBATCH --job-name=group_error
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=group_error_log.txt
#SBATCH --error=group_error_error.txt

source /home/jaschlic/venv/bin/activate
python -u group_error_analysis.py
