#!/bin/bash
#SBATCH --job-name=amber-7b
#SBATCH --partition=gpumid
#SBATCH --nodes=56
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --reservation=training

srun python main.py --n_nodes 56 --run_wandb
