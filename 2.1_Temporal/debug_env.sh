#!/bin/bash
#SBATCH --job-name=debug_env
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=debug_env_log.txt
#SBATCH --error=debug_env_error.txt

echo "=== Hostname / Node ==="
hostname

echo "=== pyvenv.cfg (zeigt urspruengliches Python) ==="
cat /home/jaschlic/venv/pyvenv.cfg

echo "=== venv/bin/python Symlink-Ziel ==="
ls -la /home/jaschlic/venv/bin/python
readlink -f /home/jaschlic/venv/bin/python

echo "=== was existiert auf DIESEM Node unter /usr/bin/python* ==="
ls -la /usr/bin/python* 2>&1

echo "=== module system? ==="
module avail python 2>&1

echo "=== vor source activate ==="
which python
which pip

source /home/jaschlic/venv/bin/activate

echo "=== nach source activate ==="
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
which python
which pip
python --version

echo "=== sys.path ==="
python -c "import sys; [print(p) for p in sys.path]"

echo "=== pandas import ==="
python -c "import pandas; print(pandas.__file__)"

echo "=== pip list (venv) ==="
python -m pip list
