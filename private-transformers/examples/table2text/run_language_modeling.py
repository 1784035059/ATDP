# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""

import json
import logging
import os

from ml_swissknife import utils
import torch
from transformers import MODEL_WITH_LM_HEAD_MAPPING, HfArgumentParser, set_seed
from transformers.models.gpt2 import GPT2Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import default_data_collator
from private_transformers import PrivacyEngine
from compiled_args import (
    DataTrainingArguments,
    ModelArguments,
    PrivacyArguments,
    TrainingArguments,
)
from misc import get_prompt_dataset, get_all_datasets, add_special_tokens
from trainer import Trainer

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

NUM_MODELS_TO_SAVE = 50


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PrivacyArguments))
    (
        model_args,
        data_args,
        training_args,
        privacy_args,
    ) = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    privacy_args: PrivacyArguments

    logger.info(f"train data: {data_args.train_data_file}")
    logger.info(f"valid data: {data_args.valid_data_file}")
    logger.info(f"eval data: {data_args.eval_data_file}")
    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use "
            f"--overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Debug mode
    if training_args.debug:
        import warnings

        warnings.filterwarnings("error")

    # Low rank models need special models!
    from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

    # Config.
    config = GPT2Config.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    config.return_dict = True
    config.tie_word_embeddings = False

    # import pdb; pdb.set_trace()
    # Tokenizer; `bos_token` and `eos_token` is the same for GPT2; both are 50256.
    tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    # Model.
    gpt2 = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    print(f"base gpt2 model: {model_args.model_name_or_path}")
    print(gpt2)

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # import pdb; pdb.set_trace()
    if not training_args.is_sdp_finetune:
        # Clone the embedding into the lm_head for better initialization.
        lm_head = gpt2.get_output_embeddings()
        embedding = gpt2.get_input_embeddings()
        lm_head.weight.data.copy_(embedding.weight.data)
        print(
            f"Cloning initial embedding into lm_head, "
            f"checking norms... \n"
            f"\tlm_head: {lm_head.weight.norm()}, embedding: {embedding.weight.norm()}"
        )
        torch.testing.assert_allclose(lm_head.weight, embedding.weight)
        del lm_head, embedding

        # Adjust tokenizer and model embeddings.
        print("adapt tokenizer to include [PAD] or other special tokens")
        print(f"before len(tokenizer) = {len(tokenizer)}")
        len_tokenizer_before = len(tokenizer)
        tokenizer = add_special_tokens(tokenizer, data_args, add_mask=model_args.add_mask)
        # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        len_tokenizer_after = len(tokenizer)
        print(f"after len(tokenizer) = {len(tokenizer)}")
        print("tokenizer.eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
        print("tokenizer.bos_token:", tokenizer.bos_token, tokenizer.bos_token_id)

        print("adapt the size of lm_head and input_embeddings to include [PAD]")
        print("use avg-based initialization")

        input_embeddings_before = gpt2.get_input_embeddings().weight
        lm_head_before = gpt2.get_output_embeddings().weight
        gpt2.resize_token_embeddings(len(tokenizer))

        input_embeddings_after = gpt2.get_input_embeddings().weight
        lm_head_after = gpt2.get_output_embeddings().weight
        print(
            f"before lm_head.weight.size() = {lm_head_before.size()}, "
            f"input_embeddings_before.size() = {input_embeddings_before.size()}"
        )
        print(
            f"after lm_head.weight.size() = {lm_head_after.size()}, "
            f"after input_embeddings_after.size() = {input_embeddings_after.size()}"
        )
        # torch.testing.assert_allclose(lm_head_before, lm_head_after[:-1])
        if len_tokenizer_after - len_tokenizer_before:
            print("pre-chunk equal for lm_head")
            torch.testing.assert_allclose(
                input_embeddings_before, input_embeddings_after[: -(len_tokenizer_after - len_tokenizer_before)]
            )
        print("pre-chunk equal for input_embeddings")
        # import pdb; pdb.set_trace()
        IGNORE_INDEX = -100
        for _i in range(len_tokenizer_after - len_tokenizer_before):
            if "<MASK>" not in tokenizer.get_added_vocab():
                lm_head_after.data[-_i] = lm_head_before.mean(dim=0)
                input_embeddings_after.data[-_i] = input_embeddings_before.mean(dim=0)
            else:
                if "abcd" in data_args.task_mode:
                    lm_head_after.data[-_i] = lm_head_before.mean(dim=0).detach().clone().to(lm_head_before.device)
                    # (lm_head_before[tokenizer.encode("mask")].detach().clone().to(lm_head_before.device))
                    input_embeddings_after.data[-_i] = (
                        input_embeddings_before.mean(dim=0).detach().clone().to(lm_head_before.device)
                    )
                    # (input_embeddings_before[tokenizer.encode("mask")].detach().clone().to(lm_head_before.device))
                    IGNORE_INDEX = tokenizer.encode("<MASK>")[0]  # len(tokenizer) - 1
                elif "wikitext2" in data_args.task_mode:
                    lm_head_after.data[-_i] = (
                        lm_head_before[tokenizer.encode("mask")].detach().clone().to(lm_head_before.device)
                    )
                    input_embeddings_after.data[-_i] = (
                        input_embeddings_before[tokenizer.encode("mask")].detach().clone().to(lm_head_before.device)
                    )
                    IGNORE_INDEX = len(tokenizer) - 1

        print("double check: ")
        print("embedding size", gpt2.get_input_embeddings().weight.size())
        print("lm_head size", gpt2.get_output_embeddings().weight.size())
    else:
        if model_args.add_mask:
            assert "<MASK>" in tokenizer.get_added_vocab()
            IGNORE_INDEX = len(tokenizer) - 1
        else:
            IGNORE_INDEX = -100

    # import pdb

    # pdb.set_trace()
    if IGNORE_INDEX != -100:
        assert tokenizer.decode(IGNORE_INDEX) == "<MASK>"
    # import pdb

    # pdb.set_trace()
    model = gpt2

    train_dataset, val_dataset, eval_dataset, data_collator = get_all_datasets(
        config=config,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        model_args=model_args,
    )
    eval_collator = default_data_collator

    # import pdb

    # pdb.set_trace()
    if not training_args.skip_generation:
        # Materialize the prompts.
        generation_stuff = dict(
            train_prompts=get_prompt_dataset(file_path=data_args.train_prompt_file, tokenizer=tokenizer),
            val_prompts=get_prompt_dataset(file_path=data_args.val_prompt_file, tokenizer=tokenizer),
            eval_prompts=get_prompt_dataset(file_path=data_args.eval_prompt_file, tokenizer=tokenizer),
        )
    else:
        generation_stuff = None
    """
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        data_args=data_args,
        privacy_args=privacy_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        generation_stuff=generation_stuff,
        ignore_index=IGNORE_INDEX,
    )    
    """
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        data_args=data_args,
        privacy_args=privacy_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        eval_data_collator=eval_collator,
        generation_stuff=generation_stuff,
        ignore_index=IGNORE_INDEX,
    )

    

    # Massage the parameters.
    if model_args.train_last_layer_only:
        model.requires_grad_(False)
        model.get_output_embeddings().requires_grad_(True)
    else:
        model.requires_grad_(True)
        if model_args.static_lm_head:
            model.get_output_embeddings().requires_grad_(False)
        if model_args.static_embedding:
            model.get_input_embeddings().requires_grad_(False)
            model.transformer.wpe.requires_grad_(False)
    params = tuple(param for param in model.parameters() if param.requires_grad)
    names = tuple(name for name, param in model.named_parameters() if param.requires_grad)
    num_trainable_params = sum(param.numel() for param in params)
    print(f"Number of trainable params: {num_trainable_params / 1e6:.4f} million")
    print(json.dumps(names, indent=4))

    # TODO: Using a single gigantic parameter group is okay only when `weight_decay` is 0.
    #   Biases and LM parameters should not be decayed perhaps even with privacy.
    optimizer = torch.optim.AdamW(
        params=params,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    #trainer.optimizer = optimizer

    # Create the lr_scheduler.
    num_update_steps_per_epoch = len(trainer.get_train_dataloader()) // trainer.args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    t_total = int(num_update_steps_per_epoch * trainer.args.num_train_epochs)

    print(f"\n\ntotal update: {t_total}\n\n")
    # import pdb

    # pdb.set_trace()
    # change save steps
    if t_total // NUM_MODELS_TO_SAVE > 0:
        _save_step = t_total // NUM_MODELS_TO_SAVE
    else:
        _save_step = 1
    training_args.save_steps = _save_step if training_args.save_all_models else -1  # _save_step
    training_args.eval_steps = _save_step

    # import pdb

    # 3) 构建 "本地" lr_scheduler (不要给 trainer.lr_scheduler)
    if training_args.lr_decay:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,   # <--- 用 "optimizer" 
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=t_total,
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    # Hacky way to set noise_multiplier.
    # import pdb

    # pdb.set_trace()
    if privacy_args.non_private:
        privacy_args.noise_multiplier = 0.0
        privacy_args.per_example_max_grad_norm = None
    else:
        actual_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        privacy_engine = PrivacyEngine(
            module=model,
            batch_size=actual_batch_size,
            sample_size=len(train_dataset),
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            epochs=training_args.num_train_epochs,
            max_grad_norm=privacy_args.per_example_max_grad_norm,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            accounting_mode=privacy_args.accounting_mode,
            ghost_clipping=privacy_args.ghost_clipping,
            detection_error_rate=privacy_args.detection_error_rate,
        )
        # Originally, these could have been null.
        print("Before attach, type(optimizer) =", type(optimizer))
        print("Before attach, has noise_multiplier attribute? =", hasattr(optimizer, "noise_multiplier"))

        privacy_args.noise_multiplier = privacy_engine.noise_multiplier
        privacy_args.target_delta = privacy_engine.target_delta

        print("privacy_args: ")
        print(json.dumps(privacy_args.__dict__, indent=4))
        privacy_engine.attach(optimizer)
        
      

    trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    model_args=model_args,
    data_args=data_args,
    privacy_args=privacy_args,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    eval_data_collator=eval_collator,
    generation_stuff=generation_stuff,
    ignore_index=IGNORE_INDEX,
    optimizers=(optimizer, lr_scheduler),    # 新增: 显式传入
)    

    # import pdb

    # pdb.set_trace()
    # Training.
    if training_args.do_train:
        all_args = {
            **training_args.__dict__,
            **data_args.__dict__,
            **model_args.__dict__,
            **privacy_args.__dict__,
        }
        utils.jdump(
            all_args,
            os.path.join(training_args.output_dir, "argparse.json"),
            default=lambda x: str(x),
        )

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        logger.info("*** Train ***")
        logger.info(
            f"Training set size: {len(train_dataset)}, "
            f"per_device_train_batch_size: {training_args.per_device_train_batch_size}, "
            f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}"
        )
        print(f"Training set size: {len(train_dataset)}, ")
        print()
        print()
        # import pdb

        # pdb.set_trace()
        # lxuechen: Especially so for the restored checkpoints. Don't resume...
        trainer.train(model_path=None)
        if training_args.save_at_last:
            trainer.save_model()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        output = trainer.evaluate(log_results=False)
        utils.jdump(
            output,
            os.path.join(training_args.output_dir, "final_results.json"),
        )

        logger.info("***** Eval results *****")
        logger.info(output)


if __name__ == "__main__":
    main()
