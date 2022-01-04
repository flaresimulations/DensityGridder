#!/bin/bash
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --job-name=density_gridder_div_example
#SBATCH --output=logs/grid_out_ref.%J
#SBATCH --error=logs/grid_err_ref.%J
#SBATCH -t 00:35
#SBATCH --ntasks 24
# #SBATCH --exclusive

module purge
module load intel_comp
module load gsl
module load openmpi
module load hdf5 

mpicc -I/usr/include/hdf5/serial -o gridder_div.x -L/usr/lib/x86_64-linux-gnu/hdf5/serial gridder_div.c -lhdf5 -lm -lgmp

mpirun gridder_div.x configs/example_config.txt

