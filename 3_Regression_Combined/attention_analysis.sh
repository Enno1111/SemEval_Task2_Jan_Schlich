#!/bin/bash
#SBATCH --job-name=attention
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/attention_log.txt
#SBATCH --error=results/logs/attention_error.txt

source /home/jaschlic/venv/bin/activate
/home/jaschlic/venv/bin/python -u attention_analysis.py
