#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --time=11:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=ablation_log.txt
#SBATCH --error=ablation_error.txt

source /home/jaschlic/venv/bin/activate
python -u ablation.py