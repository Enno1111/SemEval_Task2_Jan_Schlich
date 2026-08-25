#!/bin/bash
#SBATCH --job-name=head_analysis
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=head_analysis_log.txt
#SBATCH --error=head_analysis_error.txt

source /home/jaschlic/venv/bin/activate
/home/jaschlic/venv/bin/python -u head_analysis.py
