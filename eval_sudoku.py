import sys, os
import torch
import torch.nn
import torch.optim
import tqdm
import torchvision
from torchvision import transforms
import numpy as np
from torch.optim.swa_utils import AveragedModel
import matplotlib.pyplot as plt

from source.data.datasets.sudoku.sudoku import SudokuDataset, HardSudokuDataset
from source.models.sudoku.knet import SudokuAKOrN
from source.models.sudoku.transformer import SudokuTransformer
from source.evals.sudoku.evals import compute_board_accuracy
from source.utils import str2bool
from ema_pytorch import EMA
import argparse

if __name__ == "__main__":

    # 创建参数解析器，定义脚本支持的命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="path to the model")

    # Data loading
    parser.add_argument("--data", type=str, default="id", help="data")
    parser.add_argument("--limit_cores_used", type=str2bool, default=False)
    parser.add_argument("--cpu_core_start", type=int, default=0, help="start core")
    parser.add_argument("--cpu_core_end", type=int, default=16, help="end core")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional. Specify the root dir of the dataset. If None, use a default path set for each dataset",
    )
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)

    # General model options
    parser.add_argument("--model", type=str, default="akorn", help="model")
    parser.add_argument("--L", type=int, default=1, help="num of layers")
    parser.add_argument("--T", type=int, default=16, help="Timesteps")
    parser.add_argument("--ch", type=int, default=512, help="num of channels")
    parser.add_argument("--heads", type=int, default=8)

    # AKOrN options
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="num of random oscillator samples for each input",
    )  # 每个输入样本生成的随机振荡器采样的总次数
    parser.add_argument("--minimum_chunk", type=int, default=None)  # 每次采样时单次处理的样本数量
    parser.add_argument("--evote_type", type=str, default="last", help="last or sum")
    parser.add_argument("--gamma", type=float, default=1.0, help="step size")
    parser.add_argument("--J", type=str, default="attn", help="connectivity")
    parser.add_argument("--use_omega", type=str2bool, default=True)
    parser.add_argument("--global_omg", type=str2bool, default=True)
    parser.add_argument("--learn_omg", type=str2bool, default=False)
    parser.add_argument("--init_omg", type=float, default=0.1)
    parser.add_argument("--nl", type=str2bool, default=True)

    parser.add_argument("--speed_test", action="store_true")

    # 解析从命令行传入的实际参数
    args = parser.parse_args()

    # 启用 cuDNN 的性能优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)

    # 限制数据加载线程所使用的 CPU 核
    if args.limit_cores_used:
        def worker_init_fn(worker_id):
            os.sched_setaffinity(0, range(args.cpu_core_start, args.cpu_core_end))

    else:
        worker_init_fn = None

    if args.data == "id":
        loader = torch.utils.data.DataLoader(
            SudokuDataset(
                args.data_root if args.data_root is not None else "./data/sudoku",
                train=False,
            ),
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
        )
    elif args.data == "ood":
        loader = torch.utils.data.DataLoader(
            HardSudokuDataset(
                args.data_root if args.data_root is not None else "./data/sudoku-rrn",
                split="test",
            ),
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
        )
    else:
        raise NotImplementedError

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

    # 加载预训练模型参数，并使用 EMA 包装模型
    model = EMA(net).cuda()
    model.load_state_dict(
        torch.load(args.model_path, weights_only=True)["model_state_dict"]
    )
    model = model.ema_model
    model.eval()

    K = args.K

    corrects_vote = 0
    corrects_avg = 0
    totals = 0

    # 定义每次计算预测时使用的样本块大小，方便在大规模推理时分批处理
    minimum_chunk = args.minimum_chunk if args.minimum_chunk is not None else K

    # 推理与评估
    for i, (X, Y, is_input) in tqdm.tqdm(enumerate(loader)):
        B = X.shape[0]
        if args.model == 'akorn' and K > 1:  # Energy-based voting
            for j in range(B):
                preds = []
                es_list = []
                
                # 按块分批处理样本
                for k in range(K//minimum_chunk):
                    
                    _X = X[j : j + 1].repeat(minimum_chunk, 1, 1, 1)
                    _Y = Y[j : j + 1].repeat(minimum_chunk, 1, 1, 1)
                    _is_input = is_input[j : j + 1].repeat(minimum_chunk, 1, 1, 1)
                    _X, _Y, _is_input = (
                        _X.to(torch.int32).cuda(),
                        _Y.cuda(),
                        _is_input.cuda(),
                    )

                    with torch.no_grad():
                        pred, es = model(_X, _is_input, return_es=True)
                        preds.append(pred.detach())
                        if args.evote_type =='sum':
                            # the sum of energy values over timesteps as board correctness indicator 
                            es = torch.stack(es[-1], 0).sum(0).detach()
                        elif args.evote_type == 'last':
                            # only use the energy value of the last step 
                            es = es[-1][-1].detach()
                        es_list.append(es)
                        
                pred = torch.cat(preds, 0)
                es = torch.cat(es_list, 0)
                
                idxes = torch.argsort(es)  # minimum energy first
                pred_vote = pred[idxes[:1]].mean(0, keepdim=True)  # 选择能量最低的预测结果，并取均值作为投票结果
                pred_avg = pred.mean(0, keepdim=True)  # 所有预测结果的平均值

                _, _, board_correct_vote = compute_board_accuracy(
                    pred_vote, _Y[:1], _is_input[:1]
                )

                corrects_vote += board_correct_vote.sum().item()
                totals += board_correct_vote.numel()
            
        else:
            X, Y, is_input = X.to(torch.int32).cuda(), Y.cuda(), is_input.cuda()
            with torch.no_grad():
                pred = model(X, is_input)
            num_blanks, num_corrects, board_correct = compute_board_accuracy(pred, Y, is_input)
            corrects_vote += board_correct.sum().item()
            totals += board_correct.numel()

    # Compute mean and standard deviation across networks
    accuracy_vote = corrects_vote / totals

    print(f"Vote accuracy: {accuracy_vote:.4f}")
