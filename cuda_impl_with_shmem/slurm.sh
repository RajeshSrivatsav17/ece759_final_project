#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J smoke_simulation
#SBATCH -o output.log -e err.log
#SBATCH --gres=gpu:1
nvcc cga.cu boundary_cond.cu velocity_correction.cu main.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o out.o
./out.o