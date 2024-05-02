#!/usr/bin/env bash
#SBATCH --job-name=llt
#SBATCH --output=py_torch_test%j.log
#SBATCH --error=py_torch_test%j.err
#SBATCH --mail-user=choi@uni-hildesheim.de
#SBATCH --partition=GPU,NGPU
#SBATCH --gres=gpu:1

# Change directory to the project folder
cd /home/choi/TIMELLM

# Activate the conda environment
source activate pytorchenv

# Set the necessary variables
model_name="TimeLLM"
train_epochs=10
learning_rate=0.01
llama_layers=32

master_port=00097
num_process=8
batch_size=24
d_model=16
d_ff=32
comment='TimeLLM-ECL'

# Define variables
model_name="TimeLLM"
train_epochs=10
learning_rate=0.01
llama_layers=32

batch_size=24
comment='TimeLLM-ECL'

# Run the Python script directly with srun
srun python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_96 \
  --model $model_name \
  --data ECL \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

