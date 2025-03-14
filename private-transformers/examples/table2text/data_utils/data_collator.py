from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    PreTrainedTokenizer,
    default_data_collator,
)


InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset
and collate them into a batch, as a dictionary of Tensors.
"""
DataCollator = NewType(
    "DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]]
)


# 1) 你可以在这里定义一个高频功能词set:
HIGH_FREQ_FUNCTION_WORDS = {
    "the", "a", "an", "of", "in", "to", "and", "or", "for", "with", "on", "at",
    "by", "so", "be", "is", "are", "am", "was", "were", "have", "has", "had",
    "not", "no", "do", "does", "did", "that", "this", "it", "he", "she", "they",
    "we", "you", "me", "him", "her", "them", "us", "but", "nor", "yet",
    # ...你可以扩充更多常见功能词
}

from dataclasses import dataclass, field
from typing import List, Dict
import torch
from transformers import PreTrainedTokenizer, default_data_collator


@dataclass
class SelectiveDPDataCollator:
    tokenizer: PreTrainedTokenizer

    # 同样的属性
    current_epoch: int = 1
    sensitive_token_stats: dict = field(default_factory=dict)

    # 新增：全局缓存 { token_id -> float权重 }
    token_weights: Dict[int, float] = field(default_factory=dict)

    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch

    def update_sensitive_stats(self, new_stats: dict):
        self.sensitive_token_stats = new_stats

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 1) 先用默认collator做最基本的padding/label创建
        batch = default_data_collator(examples)

        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()

        if "input_ids" not in batch or "labels" not in batch:
            return batch

        input_ids = batch["input_ids"]
        labels    = batch["labels"]
        B, seq_len = input_ids.shape

        final_input_ids_list  = []
        final_labels_list     = []
        final_loss_weights_list = []

        for i in range(B):
            inside = False
            new_ids = []
            new_lbs = []
            new_wts = []

            for j in range(seq_len):
                tid = input_ids[i, j].item()
                lbl = labels[i, j].item()

                # 跳过 label=-100 （padding位置）
                if lbl == -100:
                    continue

                tok_str = self.tokenizer.decode([tid], clean_up_tokenization_spaces=False).lower()

                # 丢弃 <...> 标记逻辑
                if '<' in tok_str and '>' in tok_str:
                    continue
                elif '<' in tok_str:
                    inside = True
                    continue
                elif '>' in tok_str:
                    inside = False
                    continue

                # ---- 核心：决定“初始” old_w 时，先看 self.token_weights 里有没有 ----
                # 没有则按原始逻辑 (inside=1 =>1.0 / outside且非功能词=>0.2 / 功能词=>1.0)
                default_w = None
                if inside:
                    default_w = 1.0
                else:
                    if tok_str in HIGH_FREQ_FUNCTION_WORDS:
                        default_w = 1.0
                    else:
                        default_w = 0.2

                # 如果我们的缓存里已经有 token_weights[tid]，就使用它；
                # 如果没有，就用 default_w
                old_w = self.token_weights.get(tid, default_w)

                new_ids.append(tid)
                new_lbs.append(lbl)
                new_wts.append(old_w)

            if len(new_ids) == 0:
                # 保护：如果都被丢弃
                new_ids = [self.tokenizer.eos_token_id]
                new_lbs = [-100]
                new_wts = [0.0]

            final_input_ids_list.append(torch.tensor(new_ids, dtype=torch.long))
            final_labels_list.append(torch.tensor(new_lbs, dtype=torch.long))
            final_loss_weights_list.append(torch.tensor(new_wts, dtype=torch.float))

        pad_token_id = self.tokenizer.pad_token_id or 0
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            final_input_ids_list, batch_first=True, padding_value=pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            final_labels_list, batch_first=True, padding_value=-100
        )
        padded_loss_weights = torch.nn.utils.rnn.pad_sequence(
            final_loss_weights_list, batch_first=True, padding_value=0.0
        )

        batch["input_ids"] = padded_input_ids
        batch["labels"] = padded_labels
        batch["loss_weights"] = padded_loss_weights
        attention_mask = (padded_input_ids != pad_token_id).long()
        batch["attention_mask"] = attention_mask

        

        # 2) 在这里做“动态调整”
        if (
            self.current_epoch > 0 
            and self.current_epoch % 5 == 0
            and len(self.sensitive_token_stats) > 0
        ):
            B2, seq_len2 = padded_input_ids.shape

            num_tokens_total = 0
            num_tokens_changed = 0

            for b_idx in range(B2):
                for s_idx in range(seq_len2):
                    old_w = batch["loss_weights"][b_idx, s_idx].item()
                    # 统计
                    num_tokens_total += 1

                    # if (old_w > 0.2) ... 或 if abs(old_w -1.0)<1e-6
                    if abs(old_w - 1.0) < 1e-6:
                        tid = padded_input_ids[b_idx, s_idx].item()
                        tok_str = self.tokenizer.decode([tid], clean_up_tokenization_spaces=False).lower()
                        if tok_str not in HIGH_FREQ_FUNCTION_WORDS:
                            avg_prob = self.sensitive_token_stats.get(tid, None)
                            if avg_prob is not None and avg_prob < 0:
                                new_w = max(old_w - 0.1, 0.2)
                                if new_w != old_w:
                                    num_tokens_changed += 1
                                batch["loss_weights"][b_idx, s_idx] = new_w
                                # ---- 更新 self.token_weights[tid] 以备下次 epoch 使用！
                                self.token_weights[tid] = new_w

            if num_tokens_total > 0:
                ratio = 100.0 * num_tokens_changed / num_tokens_total
                print(f"[DEBUG] epoch={self.current_epoch}, changed={num_tokens_changed}, total={num_tokens_total}, ratio={ratio:.2f}%")

        else:
            # 即使在没触发调整的 epoch，也要把 current "loss_weights" 写回 token_weights
            # 这样 1.0/0.2 初始值会保留到下一轮
            B2, seq_len2 = padded_input_ids.shape
            for b_idx in range(B2):
                for s_idx in range(seq_len2):
                    w = batch["loss_weights"][b_idx, s_idx].item()
                    tid = padded_input_ids[b_idx, s_idx].item()
                    # 只要 tid != pad_token_id/...
                    if w > 0.0:  # or other condition
                        self.token_weights[tid] = w
        
        
        return batch



    
@dataclass
class DataCollatorForData2TextLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    format_mode: str = "cat"
    mlm_probability: float = 0.15

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        input_ids, labels, src, tgt, cate = zip(*examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            if self.format_mode == "cat":
                mode_input = 3
            elif self.format_mode == "peek":
                mode_input = 1
            elif self.format_mode == "nopeek":
                mode_input = 2
            elif self.format_mode == "infix":
                mode_input = 4

            # mode_input = 1 # means that we take the input again.
            # mode_input = 2 # means that we do not peek at src again.
            # mode_input = 3 # means that we look at the categories, and see the input again.

            if mode_input == 1:
                # input, batch
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
                # tgt = self._tensorize_batch(tgt)
            elif mode_input == 2:
                # nopeek.
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
            elif mode_input == 3:
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(cate)
                cate_batch, cate_attn = None, None
            elif mode_input == 4:
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)

                cate_batch = self._tensorize_batch(cate)
                cate_attn = cate_batch != self.tokenizer.pad_token_id

            labels[labels == self.tokenizer.pad_token_id] = -100  # tgt
            src_attn = src != self.tokenizer.pad_token_id  # src
            tgt_attn = batch != self.tokenizer.pad_token_id  # tgt

            if cate_batch is None:
                return {
                    "input_ids": batch,
                    "labels": labels,
                    "src_attn": src_attn,
                    "tgt_attn": tgt_attn,
                    "src": src,
                }
            else:
                return {
                    "input_ids": batch,
                    "labels": labels,
                    "src_attn": src_attn,
                    "tgt_attn": tgt_attn,
                    "src": src,
                    "cate_batch": cate_batch,
                    "cate_attn": cate_attn,
                }

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(
                examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


@dataclass
class DataCollatorForSumLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    format_mode: str = "cat"
    mlm_probability: float = 0.15

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        # print(examples[0])
        # print(len(examples))
        input_ids, labels, src, tgt = zip(*examples)
        # print(len(input_ids), len(labels), len(weights))
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:

            # print(self.format_mode)

            if self.format_mode == "peek" or self.format_mode == "cat":
                mode_input = 1
            elif self.format_mode == "nopeek":
                assert False, "should use format_mode = peek or cat."
                mode_input = 2
            elif self.format_mode == "infix":
                assert False, "should use format_mode = peek or cat."
                mode_input = 4

            # mode_input = 1 # means that we take the input again.
            # mode_input = 2 # means that we do not peek at src again.
            # mode_input = 3 # means that we look at the categories, and see the input again.

            # print(self.format_mode, mode_input)

            if mode_input == 1:
                # input, batch
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(src)

            labels[labels == self.tokenizer.pad_token_id] = -100  # tgt
            src_attn = src != self.tokenizer.pad_token_id  # src
            tgt_attn = batch != self.tokenizer.pad_token_id  # tgt

            return {
                "input_ids": batch,
                "labels": labels,
                "src_attn": src_attn,
                "tgt_attn": tgt_attn,
                "src": src,
            }

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(
                examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )