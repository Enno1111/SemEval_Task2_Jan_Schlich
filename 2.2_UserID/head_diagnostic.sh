#!/bin/bash
#SBATCH --job-name=head_diag
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=head_diag_log.txt
#SBATCH --error=head_diag_error.txt

source /home/jaschlic/venv/bin/activate
/home/jaschlic/venv/bin/python -u head_diagnostic.py
