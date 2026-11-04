# Sourced by the job scripts. Uses the group's shared venv on Kuma.
GROUP_DIR=/work/cs-471/the-girls

module purge
module load gcc python
source "$GROUP_DIR/venv/bin/activate"

export OMP_NUM_THREADS=8
