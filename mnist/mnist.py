#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/pytorch/opacus/blob/a8fecaf9327d6cbfa73d2d026cc414e6c12ddead/examples/mnist.py
"""
Runs MNIST training with differential privacy.

"""
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from privacy_engine_with_filter import PublicEngine, PrivacyEngineWithFilter
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from copy import deepcopy

from torch.utils.data.dataset import Dataset
import random

MIDDLE = 14
OFFSET = 4
# (8*8)/(28*28) = 8.2%
NORMALIZED_BLACK = -0.42421296
SEED = 123456
GRAD_TO_SAVE = []


def policy_func(ex):
    if len(ex.shape) == 4:
        ex[:, 0, (MIDDLE - OFFSET) : (MIDDLE + OFFSET), (MIDDLE - OFFSET) : (MIDDLE + OFFSET)] = NORMALIZED_BLACK
    elif len(ex.shape) == 3:
        ex[0, (MIDDLE - OFFSET) : (MIDDLE + OFFSET), (MIDDLE - OFFSET) : (MIDDLE + OFFSET)] = NORMALIZED_BLACK
    return ex


class NormalizedDataset(Dataset):
    def __init__(self, train_dataset):
        self.examples = self.normalize_traindata(train_dataset)

    def normalize_traindata(self, train_dataset):
        "apply policy function to the train dataset"
        examples = []
        for data, target in train_dataset:
            ex = data.clone().detach()
            ex = policy_func(ex)
            examples.append((ex, target))
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SimpleSampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        # self.conv2 = nn.Conv2d(16, 32, 4, 2)
        # self.fc1 = nn.Linear(32 * 4 * 4, 32)
        # self.fc2 = nn.Linear(32, 10)
        self.fc1 = nn.Linear(28 * 28, 10, bias=False)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        # x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        # x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        # x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        # x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        # x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        # x = F.relu(self.fc1(x))  # -> [B, 32]
        # x = self.fc2(x)  # -> [B, 10]
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return x

    def name(self):
        return "SimpleSampleConvNet"


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


def train(args, model, device, train_loader, optimizer, epoch, test_loader=None):
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        model.train()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # import pdb

        # pdb.set_trace()
        # model.fc1.weight.grad.data.norm(2)
        optimizer.step()
        grad_to_save = model.fc1.weight.grad.data.clone().detach().cpu().numpy()
        GRAD_TO_SAVE.append(grad_to_save)
        losses.append(loss.item())
        if args.test_every_batch:
            test(args, model, device, test_loader)

    if not args.disable_dp:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def train_with_filter(args, model, device, train_loader, optimizer, epoch, test_loader=None, public_train_loader=None):
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        model.train()
        import pdb

        pdb.set_trace()
        normalized_data = data.clone().detach()
        normalized_data = policy_func(normalized_data)
        normalized_data, data, target = normalized_data.to(device), data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        # import pdb

        # pdb.set_trace()
        # model.fc1.weight.grad.data.norm(2)
        optimizer.step()
        grad_to_save = model.fc1.weight.grad.data.clone().detach().cpu().numpy()
        GRAD_TO_SAVE.append(grad_to_save)
        losses.append(loss.item())
        if args.test_every_batch:
            test(args, model, device, test_loader)

    if not args.disable_dp:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(args, model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "-sr",
        "--sample-rate",
        type=float,
        default=0.001,
        metavar="SR",
        help="sample rate used for batch construction (default: 0.001)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: .1)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model (default: false)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
        "--normalize",
        "-nor",
        action="store_true",
        default=False,
        help="normalize the dataset",
    )
    parser.add_argument(
        "--use-filter",
        "-f",
        action="store_true",
        default=False,
        help="use kf filter",
    )
    parser.add_argument(
        "--use_public_engine",
        "-pub",
        action="store_true",
        default=False,
        help="use public engine",
    )
    parser.add_argument(
        "--test_every_batch",
        "-t",
        action="store_true",
        default=False,
        help="test every batch",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    kwargs = {"num_workers": 1, "pin_memory": True}

    make_deterministic(SEED)

    if args.secure_rng:
        try:
            import torchcsprng as prng
        except ImportError as e:
            msg = (
                "To use secure RNG, you must install the torchcsprng package! "
                "Check out the instructions here: https://github.com/pytorch/csprng#installation"
            )
            raise ImportError(msg) from e

        generator = prng.create_random_device_generator("/dev/urandom")

    else:
        generator = None

    train_dataset = datasets.MNIST(
        args.data_root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ]
        ),
    )

    if args.normalize:
        train_dataset = NormalizedDataset(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        generator=generator,
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(train_dataset),
            sample_rate=args.sample_rate,
            generator=generator,
        ),
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )
    run_results = []
    for _ in range(args.n_runs):
        model = SimpleSampleConvNet().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        import pdb

        pdb.set_trace()
        if not args.disable_dp:
            if not args.use_filter:
                privacy_engine = PrivacyEngine(
                    model,
                    sample_rate=args.sample_rate,
                    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                    noise_multiplier=args.sigma,
                    max_grad_norm=args.max_per_sample_grad_norm,
                    secure_rng=args.secure_rng,
                    seed=SEED,
                )
            else:
                privacy_engine = PrivacyEngineWithFilter(
                    model,
                    sample_rate=args.sample_rate,
                    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                    noise_multiplier=args.sigma,
                    max_grad_norm=args.max_per_sample_grad_norm,
                    secure_rng=args.secure_rng,
                    seed=SEED,
                )
                privacy_engine.setup_filter()
            privacy_engine.attach(optimizer)
        else:
            # use public engine to get the clipped true grads
            if args.use_public_engine:
                privacy_engine = PublicEngine(
                    model,
                    sample_rate=args.sample_rate,
                    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                    noise_multiplier=args.sigma,
                    max_grad_norm=args.max_per_sample_grad_norm,
                    secure_rng=args.secure_rng,
                    seed=SEED,
                )
                privacy_engine.attach(optimizer)
        for epoch in range(1, args.epochs + 1):
            train_with_filter(args, model, device, train_loader, optimizer, epoch, test_loader=test_loader)
            test(args, model, device, test_loader)
        run_results.append(test(args, model, device, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

    repro_str = (
        f"{model.name()}_{args.lr}_{args.sigma}_" f"{args.max_per_sample_grad_norm}_{args.sample_rate}_{args.epochs}"
    )
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")
        save_dir = f"grads_normalize={args.normalize}_dp={not args.disable_dp}_sample-rate={args.sample_rate}_epoch={args.epochs}_pubengine={args.use_public_engine}_filter={args.use_filter}_seed={SEED}.pt"
        torch.save(
            GRAD_TO_SAVE,
            save_dir,
        )
        print(save_dir)


if __name__ == "__main__":
    main()
