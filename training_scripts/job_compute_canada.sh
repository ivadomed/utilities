#!/bin/bash
#SBATCH --account=def-jcohen
#SBATCH --job-name=job1     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1-06:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/<your-user-name>/code/nnunet-v2/jobs/sciseg-v2/region-based/outs/%x_%A_v2.out
#SBATCH --error=/home/<your-user-name>/code/nnunet-v2/jobs/sciseg-v2/region-based/errs/%x_%A_v2.err
#SBATCH --mail-user=<your-email-id>     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=begin,end

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# load the required modules
echo "Loading modules ..."
module load python/3.10.13 cuda/12.2    # TODO: might differ depending on the python and cuda version you have

# activate environment
echo "Activating environment ..."
source /home/$(whoami)/envs/venv_nnunet/bin/activate        # TODO: update to match the name of your environment

# Run the model
bash <path/to/run_nnunet_compute_canada/script>