import torch
import sys, os
import tqdm
import argparse

from source.models.sudoku.transformer import SudokuTransformer

from source.training_utils import save_checkpoint, save_model
from source.data.datasets.sudoku.sudoku import SudokuDataset, HardSudokuDataset, convert_onehot_to_int
from source.models.sudoku.knet import SudokuAKOrN
from source.utils import str2bool
from source.sudoku_visualizer import SudokuVisualizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ema_pytorch import EMA

from torch.utils.tensorboard import SummaryWriter


def apply_threshold(model, threshold):
    with torch.no_grad():
        for param in model.parameters():
            # Set elements with absolute values less than threshold to zero,
            # removing insignificant weights for sparsity optimization
            param.data = torch.where(
                param.abs() < threshold, torch.tensor(0.0), param.data
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()  # Argument parser for command-line arguments

    # Add command-line arguments
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--beta", type=float, default=0.995, help="EMA decay")
    parser.add_argument(
        "--clip_grad_norm", type=float, default=1.0, help="clip gradient norm"
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="save checkpoint every specified epochs",
    )  # Save model checkpoint every specified epochs
    parser.add_argument("--eval_freq", type=int, default=10, help="evaluation frequency")  # Perform evaluation every specified epochs

    # Data loading parameters
    parser.add_argument("--limit_cores_used", type=str2bool, default=False)
    parser.add_argument("--cpu_core_start", type=int, default=0, help="start core")
    parser.add_argument("--cpu_core_end", type=int, default=16, help="end core")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional. Specify the root directory of the dataset. If None, use a default path set for each dataset",
    )
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)

    # General model options
    parser.add_argument("--model", type=str, default="akorn", help="model")
    parser.add_argument("--L", type=int, default=1, help="number of layers")
    parser.add_argument("--T", type=int, default=16, help="timesteps")
    parser.add_argument("--ch", type=int, default=512, help="number of channels")
    parser.add_argument("--heads", type=int, default=8)

    # AKOrN model parameters
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0, help="step size")
    parser.add_argument("--J", type=str, default="attn", help="connectivity type")
    parser.add_argument("--use_omega", type=str2bool, default=True)
    parser.add_argument("--global_omg", type=str2bool, default=True)
    parser.add_argument("--learn_omg", type=str2bool, default=False)
    parser.add_argument("--init_omg", type=float, default=0.1)
    parser.add_argument("--nl", type=str2bool, default=True)

    parser.add_argument("--speed_test", action="store_true")  # Whether to measure time cost per iteration during training

    args = parser.parse_args()  # Parse command-line arguments

    print("Experiment name: ", args.exp_name)

    # Enable torch.backends.cudnn optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    
    # Set random seed for reproducibility
    if args.seed is not None:
        import random
        import numpy as np

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Control CPU core allocation for multi-threaded data loading
    def worker_init_fn(worker_id):
        os.sched_setaffinity(0, range(args.cpu_core_start, args.cpu_core_end))

    # Load dataset from the specified path
    if args.data_root is not None:
        rootdir = args.data_root
    else:
        rootdir = "./data/sudoku"
    
    # Load training data, set batch size and data loading method
    trainloader = torch.utils.data.DataLoader(
        SudokuDataset(rootdir, train=True),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )
    # Load test data, set batch size and data loading method
    testloader = torch.utils.data.DataLoader(
        SudokuDataset(rootdir, train=False),
        batch_size=100,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    jobdir = f"runs/{args.exp_name}/"
    writer = SummaryWriter(jobdir)

    # Only compute digit-wise accuracy
    from source.evals.sudoku.evals import compute_board_accuracy
    def compute_acc(net, loader):
        net.eval()
        correct = 0
        total = 0
        correct_input = 0
        total_input = 0
        for X, Y, is_input in loader:
            X, Y, is_input = X.to(torch.int32).cuda(), Y.cuda(), is_input.cuda()

            with torch.no_grad():
                out = net(X, is_input)
            
            _, _, board_accuracy = compute_board_accuracy(out, Y, is_input)  # Number of completely correct boards per batch
            correct += board_accuracy.sum().item()
            total += board_accuracy.shape[0]
           
            # Digit-wise input accuracy
            out = out.argmax(dim=-1)
            Y = Y.argmax(dim=-1)
            mask = (1 - is_input).view(out.shape)
            correct_input += ((1 - mask) * (out == Y)).sum().item()
            total_input += (1 - mask).sum().item()

        acc = correct / total  # Overall board accuracy (probability of a completely correct board)
        input_acc = correct_input / total_input  # Accuracy on input regions (probability of correct digits in given positions)
        return acc, input_acc, (total, correct), (total_input, correct_input)

    # Model selection
    if args.model == "akorn":
        print(
            f"n: {args.N}, ch: {args.ch}, L: {args.L}, T: {args.T}, type of J: {args.J}"
        )
        net = SudokuAKOrN(
            n=args.N,
            ch=args.ch,
            L=args.L,
            T=args.T,
            gamma=args.gamma,
            J=args.J,
            use_omega=args.use_omega,
            global_omg=args.global_omg,
            init_omg=args.init_omg,
            learn_omg=args.learn_omg,
            nl=args.nl,
            heads=args.heads,
        )
    elif args.model == "itrsa":
        net = SudokuTransformer(
            ch=args.ch,
            blocks=args.L,
            heads=args.heads,
            mlp_dim=args.ch * 2,
            T=args.T,
            gta=False,
        )
    else:
        raise NotImplementedError

    net.cuda()
    
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    # Define optimizer, EMA updater, and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Exponential Moving Average (EMA) smooths model parameters, improving generalization.
    ema = EMA(net, beta=args.beta, update_every=10, update_after_step=100)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
