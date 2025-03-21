#!/bin/bash

output_dir=${1}
data_dir=${2}
task_mode=${3}
model_name_or_path=${4:-"gpt2"} # One of distilgpt2, gpt2, gpt2-medium, gpt2-large
target_epsilon=${5:-"8"}
# cache_dir=${6}
ghost_clipping=${6:-"yes"} # Fill 'no' to turn this off.
non_private=${7:-"no"}
is_sdp_finetune=${8:-"no"}
num_train_epochs=${9:-"3"}
add_canary=${10:-"yes"}
miss_canary=${11:-"no"}
canary_times=${12:-"10"}
learning_rate=${13:-"5e-4"}
gradient_accumulation_steps=${14:-"512"}
add_mask=${15:-"yes"}
detection_error_rate=${16:-"-1"}
save_all_models=${17:-"no"}
use_different_canary=${18:-"no"}
num_canary_to_mask=${19:-"0"}
per_example_max_grad_norm=${20:-"0.1"}
lr_decay=${21:-"no"}
accounting_mode=${22:-"rdp_cks"}

if [[ ${task_mode} == "e2e" ]]; then
  # t_total=410
  data_dir="${data_dir}/data/e2e_data"
  target_delta=8e-6
  num_train_epochs=10
  learning_rate=2e-3
  max_seq_len=100
  # batch size
  per_device_eval_batch_size=10
  per_device_train_batch_size=16
  gradient_accumulation_steps=64
  block_size=-1
  eval_epochs=2
  max_eval_batches=100
  save_steps=100
  eval_steps=100
  skip_generation="yes"
elif [[ ${task_mode} == "dart" ]]; then
  # 885 t_total updates
  target_delta=1e-5
  data_dir="${data_dir}/data/dart"
  num_train_epochs=15 # Approximately same number of updates.
  learning_rate=5e-4  # Lower learning rate for stability in large models.
  max_seq_len=120
  # batch size
  per_device_eval_batch_size=10
  per_device_train_batch_size=16
  gradient_accumulation_steps=64
  block_size=-1
  eval_epochs=2
  max_eval_batches=100
  save_steps=100
  eval_steps=100
  skip_generation="yes"
elif [[ ${task_mode} == "wikitext2"* ]]; then
  # if [[ ${task_mode} == "wikitext2" ]]; then
  #   data_dir="${data_dir}/wikitext-2-raw/"
  # # wiki entity
  # elif [[ ${task_mode} == "wikitext2-delex-person" ]]; then
  #   data_dir="${data_dir}/wiki_entity_person-3.3/"
  # elif [[ ${task_mode} == "wikitext2-delex-medium" ]]; then
  #   data_dir="${data_dir}/wiki_entity_person_org_date_gpe-11.3/"
  # elif [[ ${task_mode} == "wikitext2-delex-high" ]]; then
  #   data_dir="${data_dir}/wiki_entity_all-16.4/"
  # # wiki contextual
  # elif [[ ${task_mode} == "wikitext2-delex-no_pronoun" ]]; then
  #   data_dir="${data_dir}/wiki_contextual_no_pronoun-33.7/"
  # elif [[ ${task_mode} == "wikitext2-delex-default" ]]; then
  #   data_dir="${data_dir}/wiki_contextual_default-34.8/"
  # elif [[ ${task_mode} == "wikitext2-delex-root" ]]; then
  #   data_dir="${data_dir}/wiki_contextual_root-39.1/"
  # elif [[ ${task_mode} == "wikitext2-delex-SRL" ]]; then
  #   data_dir="${data_dir}/wiki_contextual_SRL-45.0/"
  # # abcd
  # elif [[ ${task_mode} == "wikitext2-abcd" ]]; then
  #   data_dir="${data_dir}/abcd/abcd_original/"
  # elif [[ ${task_mode} == "wikitext2-abcd-delex" ]]; then
  #   data_dir="${data_dir}/abcd/abcd_delex/"
  # fi
  # len(trainer.get_train_dataloader()) = 1181
  target_delta=1e-6
  num_train_epochs=${num_train_epochs} # Approximately same number of updates.
  learning_rate=${learning_rate}  # Lower learning rate for stability in large models.
  max_seq_len=1_000_000
  per_device_eval_batch_size=2
  per_device_train_batch_size=2 # 2 is the largest for multilingual
  gradient_accumulation_steps=${gradient_accumulation_steps}
  block_size=1024
  skip_generation="yes"
  eval_epochs=1
  max_eval_batches=-1
  save_steps=1000
  eval_steps=1000
else
    echo "Unknown task: ${task_mode}"
    exit 1
fi

# Arguments in the last two lines are the most important.
python run_language_modeling.py \
  --output_dir ${output_dir} --overwrite_output_dir \
  --task_mode ${task_mode} \
  --model_name_or_path ${model_name_or_path} \
  --tokenizer_name ${model_name_or_path} \
  --do_train --do_eval \
  --line_by_line \
  --save_steps ${save_steps} --save_total_limit 1 --save_at_last no \
  --logging_dir ${output_dir} --logging_steps -1 \
  --seed 0 \
  --eval_steps ${eval_steps} --eval_epochs ${eval_epochs} \
  --max_eval_batches ${max_eval_batches} \
  --evaluation_strategy steps --evaluate_before_training "yes" --evaluate_during_training "yes" \
  --max_generations 9223372036854775807 --max_generations_train 10 --max_generations_valid 9223372036854775807 \
  --max_train_examples 9223372036854775807 --max_valid_examples 9223372036854775807 --max_eval_examples 9223372036854775807 \
  --data_folder ${data_dir} --max_seq_len ${max_seq_len} --format_mode cat \
  --per_example_max_grad_norm ${per_example_max_grad_norm} --target_delta ${target_delta} --target_epsilon ${target_epsilon} \
  --learning_rate ${learning_rate} --lr_decay ${lr_decay} --num_train_epochs ${num_train_epochs} \
  --per_device_eval_batch_size ${per_device_eval_batch_size} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --skip_generation ${skip_generation} \
  --add_mask ${add_mask} \
  --detection_error_rate ${detection_error_rate} \
  --save_all_models ${save_all_models} \
  --use_different_canary ${use_different_canary} \
  --num_canary_to_mask ${num_canary_to_mask} \
  --block_size ${block_size} \
  --is_sdp_finetune ${is_sdp_finetune} \
  --add_canary ${add_canary} \
  --miss_canary ${miss_canary} \
  --canary_times ${canary_times} \
  --non_private ${non_private} \
  --ghost_clipping ${ghost_clipping} --accounting_mode ${accounting_mode}
