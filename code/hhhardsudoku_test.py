import torch
import numpy as np
from eval_sudoku import *

if __name__ == "__main__":
    # Define the Sudoku puzzle as a 9x9 grid
    sudoku_grid = np.array([
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ])

    # Convert the Sudoku grid into a one-hot encoded tensor of shape (1, 9, 9, 9)
    onehot_tensor = np.zeros((1, 9, 9, 9), dtype=np.float32)
    is_input = np.zeros((1,9,9,1), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            if sudoku_grid[i, j] > 0:
                onehot_tensor[0, i, j, sudoku_grid[i, j] - 1] = 1
                is_input[0, i, j, 0] = 1

    device = "cuda"
    # Convert to PyTorch tensor
    onehot_tensor_torch = torch.tensor(onehot_tensor).to(device)
    is_input = torch.tensor(is_input).to(device)

    # Create argument parser to define supported command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Path to the model")

    # Data loading
    parser.add_argument("--data", type=str, default="id", help="Dataset")
    parser.add_argument("--limit_cores_used", type=str2bool, default=False)
    parser.add_argument("--cpu_core_start", type=int, default=0, help="Starting CPU core")
    parser.add_argument("--cpu_core_end", type=int, default=16, help="Ending CPU core")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional. Specify the root directory of the dataset. If None, use a default path for each dataset",
    )
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)

    # General model options
    parser.add_argument("--model", type=str, default="akorn", help="Model type")
    parser.add_argument("--L", type=int, default=1, help="Number of layers")
    parser.add_argument("--T", type=int, default=16, help="Timesteps")
    parser.add_argument("--ch", type=int, default=512, help="Number of channels")
    parser.add_argument("--heads", type=int, default=8)

    # AKOrN options
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="Number of random oscillator samples per input",
    )  # Total number of random oscillator samples generated per input
    parser.add_argument("--minimum_chunk", type=int, default=None)  # Number of samples processed per batch
    parser.add_argument("--evote_type", type=str, default="last", help="Last or sum")
    parser.add_argument("--gamma", type=float, default=1.0, help="Step size")
    parser.add_argument("--J", type=str, default="attn", help="Connectivity")
    parser.add_argument("--use_omega", type=str2bool, default=True)
    parser.add_argument("--global_omg", type=str2bool, default=True)
    parser.add_argument("--learn_omg", type=str2bool, default=False)
    parser.add_argument("--init_omg", type=float, default=0.1)
    parser.add_argument("--nl", type=str2bool, default=True)

    parser.add_argument("--speed_test", action="store_true")

    # Parse command-line arguments
    args = parser.parse_args()
    
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
    
    # Load pre-trained model parameters and wrap it with EMA
    model = EMA(net).cuda()
    model.load_state_dict(
        torch.load('runs/sudoku_akorn/ema_99.pth', weights_only=True)["model_state_dict"], strict=False
    )
    
    model = model.ema_model

    pred = model(onehot_tensor_torch, is_input, return_es=False)
    print((pred.argmax(dim=-1) + 1).cpu().numpy())
