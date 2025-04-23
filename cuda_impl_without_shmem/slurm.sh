#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J smoke_simulation
#SBATCH -o output.log -e err.log
#SBATCH --gres=gpu:1
make smoke
./smoke