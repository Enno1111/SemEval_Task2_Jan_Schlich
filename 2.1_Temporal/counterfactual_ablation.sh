#!/bin/bash
#SBATCH --job-name=counterfactual
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/counterfactual_log.txt
#SBATCH --error=results/logs/counterfactual_error.txt

source /home/jaschlic/venv/bin/activate
/home/jaschlic/venv/bin/python -u counterfactual_ablation.py
