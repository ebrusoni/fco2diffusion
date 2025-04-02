#SBATCH -c 1                            # 1 cores (for an MPI job, otherwise you would use -c 24)
#SBATCH --time 04:00:00                 # 4-hour run-time
#SBATCH --mem-per-cpu=4000              # 4000 MB per core
#SBATCH --gpus=1                         # 1 GPU
#SBATCH -o training.out                  # Write the log output to training.out
#SBATCH -J training                      # job name

module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy

pip install --upgrade pip
pip install -r requirements.txt

python esub.py


