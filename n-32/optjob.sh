#!/bin/bash
#SBATCH --partition=hmemq
#SBATCH --job-name=optcherryn32
#SBATCH --cpus-per-task=1
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=logs/optimise_kj_%j.out
#SBATCH --error=logs/optimise_kj_%j.err

/gpfs01/home/psymf9/miniconda3/envs/choice_rf/bin/python /gpfs01/home/psymf9/choice-rf-rt/optimisation/n-32/optimise.py