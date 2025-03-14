"""
python attacks/canary_insertion/canary_insertion.py -bs 256 --checkpoint /local/data/wyshi/sdp_transformers/private-transformers/examples/table2text/output/wiki/wiki_contextual_default_mask_consec-34.8/not_missed/SDP/best --outputf attacks/canary_insertion/wiki_not_missed_default.csv --cuda cuda:7
"""

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils

import argparse
import string

import torch
import torch.nn as nn

import math
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json

from transformers.models.gpt2 import GPT2Tokenizer

from scipy.integrate import quad
import scipy.stats
import numpy as np

import pandas as pd
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

"""
about 5 mins for 6-digit canary " My ID is 341752." for one model
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210409/185850 --outputf attacks/canary_insertion/nodp_10insertion.csv
"""


class CanaryDataset(Dataset):
    def __init__(self, canary, canary_list, tokenizer):
        self.canary = canary
        self.canary_list = canary_list
        self.data = self.build_data()
        self.tokenizer = tokenizer

    def build_data(self):
        texts = []
        encoded_texts = []
        for i in tqdm(range(10), desc="building the dataset"):
            for j in range(10):
                for k in range(10):
                    for l in range(10):
                        for m in range(10):
                            for n in range(10):
                                # for o in range(10):
                                # for p in range(10):
                                #     for q in range(10):
                                text = f"My ID is {i}{j}{k}{l}{m}{n}."
                                texts.append(text)
                                encoded_texts.append(tokenizer.encode(text))
        assert self.canary in texts
        for canary in self.canary_list:
            assert canary in texts
        return list(zip(texts, encoded_texts))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate(self, unpacked_data):
        return unpacked_data


def get_exposure(model, dataloader, save_json=None, multiple_canaries=False):
    ###############################################################################
    # calculate ppl
    ###############################################################################
    def calculate_exposure(canary_rank):
        return math.log(TOTAL_CANDIDATES, 2) - math.log(canary_rank, 2)

    ppls = {}
    for batch in tqdm(dataloader, desc="batch in get_exposure"):
        batch_text = list(map(lambda x: x[0], batch))
        batch_encoded_text = list(map(lambda x: x[1], batch))
        batch_ppl = utils.calculate_ppl_gpt2(
            batch_encoded_text,
            model,
            device,
            PAD_TOKEN_ID,
        )
        # import pdb; pdb.set_trace()
        ppls.update(dict(zip(batch_text, batch_ppl)))

    print("sorting...")
    sorted_ppls = {k: (i + 1, v) for i, (k, v) in enumerate(sorted(ppls.items(), key=lambda item: item[1]))}
    N = len(sorted_ppls)
    if multiple_canaries:
        canary_rank, canary_ppl, canary_exposure = [], [], []
        for canary in CANARY_LIST:
            cur_canary_rank, cur_canary_ppl = sorted_ppls[canary]
            canary_rank.append(cur_canary_rank)
            canary_ppl.append(cur_canary_ppl)
            canary_exposure.append(calculate_exposure(cur_canary_rank))
    else:
        canary_rank, canary_ppl = sorted_ppls[CANARY]
        canary_exposure = calculate_exposure(canary_rank)

    if args.debug:
        with open(args.json_dir, "w") as fh:
            json.dump(sorted_ppls, fh)

    print("canary exposure")
    print(canary_exposure)
    print("canary ranking")
    print(canary_rank)

    return canary_exposure, canary_rank, canary_ppl


def load_model_and_tokenizer(model_path, dvice):
    config = GPT2Config.from_pretrained(
        model_path,
    )
    config.return_dict = True
    config.tie_word_embeddings = False

    tokenizer = GPT2Tokenizer.from_pretrained(
        model_path,
    )
    model = GPT2LMHeadModel.from_pretrained(
        model_path,
        config=config,
    ).to(device)

    model.eval()
    return model, tokenizer


def get_model_metrics(model_path):
    with open(os.path.join(model_path, "log_history.json")) as fh:
        log_history = json.load(fh)

    result = log_history[-1]
    metrics = {
        "valid_ppl": result["val"]["model"]["ppl"],
        "test_ppl": result["eval"]["model"]["ppl"],
    }
    print(metrics["valid_ppl"])
    metrics.update({k: v for k, v in result.items() if k not in ["lr", "eval", "train", "val"]})

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Wikitext-2 Language Model")

    # Model parameters.
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        help="model checkpoint to use",
    )
    parser.add_argument("--outputf", type=str, help="output file for generated text")
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--batch_size", "-bs", type=int, default=256, help="batch size")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="use CUDA")
    parser.add_argument(
        "--missed", action="store_true", help="calculate the exposure for the missed canary"  # default=False,
    )
    parser.add_argument(
        "--multiple_canaries",
        action="store_true",
        help="calculate the exposure for the multiple canaries",  # default=False,
    )
    parser.add_argument(
        "--data_type",
        type=str.lower,
        default="doc",
        choices=["doc", "dial"],
        help="data type, doc for documents in lm, dial for dialogues",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--json_dir", type=str)
    args = parser.parse_args()

    if not os.path.exists("/".join(args.outputf.split("/")[:-1])):
        os.makedirs("/".join(args.outputf.split("/")[:-1]))
    print(f"output will be saved to {args.outputf}")
    assert not os.path.isfile(args.outputf)
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device(args.cuda)

    ###############################################################################
    # load path
    ###############################################################################
    if ("best" not in args.checkpoint) and ("checkpoint" not in args.checkpoint):
        paths = sorted(Path(args.checkpoint).iterdir(), key=os.path.getmtime)
        paths = [p for p in paths if os.path.isdir(p)]
        # paths = [p for p in paths if "best" not in str(p) and int(str(p).split("-")[-1]) < 650]
    else:
        paths = [args.checkpoint]

    # to get the tokenizer first to create the dataset
    _, tokenizer = load_model_and_tokenizer(str(paths[0]), device)
    PAD_TOKEN_ID = tokenizer.bos_token_id
    ###############################################################################
    # load data
    ###############################################################################
    CANARY = "My ID is 341752."
    CANARY_LIST = [
        "My ID is 341752.",
        "My ID is 151401.",
        "My ID is 343188.",
        "My ID is 480519.",
        "My ID is 203195.",
        "My ID is 893752.",
        "My ID is 726839.",
        "My ID is 861710.",
        "My ID is 135470.",
        "My ID is 589883.",
    ]
    CANARY_CORPUS = CanaryDataset(CANARY, CANARY_LIST, tokenizer)
    TOTAL_CANDIDATES = 1_000_000

    dataloader = DataLoader(
        dataset=CANARY_CORPUS, shuffle=False, batch_size=args.batch_size, collate_fn=CANARY_CORPUS.collate
    )

    ###############################################################################
    # Load model
    ###############################################################################
    # exposures, ranks, canary_ppls, model_ppls, model_accs = [], [], [], [], []
    records = []
    for model_path in tqdm(paths):
        model_path = str(model_path)
        model, tokenizer = load_model_and_tokenizer(model_path, device)
        canary_exposure, canary_rank, canary_ppl = get_exposure(
            model, dataloader, save_json=None, multiple_canaries=args.multiple_canaries
        )
        model_metrics = get_model_metrics(model_path)
        if args.multiple_canaries:
            model_metrics.update(
                {
                    f"canary_exposure_{i}": exp
                    for i, exp in enumerate(
                        canary_exposure,
                    )
                }
            )
            model_metrics.update(
                {
                    f"canary_rank_{i}": rank
                    for i, rank in enumerate(
                        canary_rank,
                    )
                }
            )
            model_metrics.update({f"canary_ppl_{i}": ppl for i, ppl in enumerate(canary_ppl)})
        else:
            model_metrics.update(
                {"canary_exposure": canary_exposure, "canary_rank": canary_rank, "canary_ppl": canary_ppl}
            )
        records.append(model_metrics)
    # records = sorted(records, key = lambda x: x[0])
    records = pd.DataFrame(
        records,
    )

    records.to_csv(args.outputf, index=None)
