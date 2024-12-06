#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -A m4680

module load python
conda activate simsopt

# python3 ./gpu_testing/gpu_tracing.py 
# python ./example_boozer.py
python ./example_boozer_alex.py
# python ./gpu_testing/test_interpolant.py

# srun -n 1  python3 ./examples/1_Simple/tracing_particle.py
# srun -n 2  python3 ./examples/1_Simple/tracing_particle.py
# srun -n 4  python3 ./examples/1_Simple/tracing_particle.py
# srun -n 8  python3 ./examples/1_Simple/tracing_particle.py
# srun -n 16 python3 ./examples/1_Simple/tracing_particle.py
# srun -n 32 python3 ./examples/1_Simple/tracing_particle.py
# srun -n 64 python3 ./examples/1_Simple/tracing_particle.py