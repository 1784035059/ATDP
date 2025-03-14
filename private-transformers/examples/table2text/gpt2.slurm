#!/bin/bash
#SBATCH --job-name=TestENV
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:l40:1
#SBATCH --account=a_singh
#SBATCH -o slurm-%j.output
#SBATCH -e slurm-%j.error

echo ">>> [DEBUG] Starting job..."

# 1. 清理模块，避免冲突
module purge

# 2. 加载 Anaconda 模块
module load anaconda3/2022.05
echo ">>> [DEBUG] Loaded anaconda3/2022.05."

# 3. 让当前 Shell 具备 conda activate 功能
eval "$(conda shell.bash hook)"

# 4. 激活 lm_privacy2
conda activate lm_privacy2
echo ">>> [DEBUG] Activated conda env: lm_privacy2"

# 5. Debug 看是否切到 lm_privacy2
echo ">>> [DEBUG] which python:"
which python

echo ">>> [DEBUG] python -V:"
python -V

echo ">>> [DEBUG] conda list | grep ml_swissknife:"
conda list | grep ml_swissknife

echo ">>> [DEBUG] python -c 'import ml_swissknife':"
python -c "import ml_swissknife" && echo ">>> ml_swissknife import success!" || echo ">>> ml_swissknife import failed!"

# 6. 运行你的脚本
echo ">>> [DEBUG] Start actual training..."
#CUDA_VISIBLE_DEVICES=0 bash run.sh \./output/wikitext2_experiment `#output_dir` \../../../data/wikitext-2-raw `#data_dir` \wikitext2 `#task_mode` \./output/wikitext2_experiment/checkpoint-800 `#model_name_or_path` \3 `#target_ep
#silon` \yes `#ghost_clipping` \no `#non_private` \yes `#is_sdp_finetune` \50 `#num_train_epochs` \yes `#add_canary` \yes `#miss_canary` \10 `#canary_times` \5e-05 `#learning_rate` \128 `#gradient_accumulation_steps` \no `#add_mask`
CUDA_VISIBLE_DEVICES=0 bash run.sh ./output/wikitext2_experiment/switch-entity-nomissed-add ../../../data/wikitext-2-raw wikitext2 ./output/wikitext2_experiment/nodpadd/best 3 yes no yes 100 yes no 10 3.7e-05 128 no -1 no no 0 0.1 yes


echo ">>> [DEBUG] Job finished."

