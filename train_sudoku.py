import torch
import sys, os
import tqdm
import argparse

from source.models.sudoku.transformer import SudokuTransformer

from source.training_utils import save_checkpoint, save_model
from source.data.datasets.sudoku.sudoku import SudokuDataset, HardSudokuDataset
from source.models.sudoku.knet import SudokuAKOrN
from source.utils import str2bool
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ema_pytorch import EMA

from torch.utils.tensorboard import SummaryWriter


def apply_threshold(model, threshold):
    with torch.no_grad():
        for param in model.parameters():
            # 将绝对值小于threshold的元素设置为0，在稀疏性优化中去除不重要的权重
            param.data = torch.where(
                param.abs() < threshold, torch.tensor(0.0), param.data
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()  # 用于解析命令行参数

    # 添加命令行参数
    parser.add_argument("--exp_name", type=str, help="expname")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("--epochs", type=int, default=100, help="num of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="lr")
    parser.add_argument("--beta", type=float, default=0.995, help="ema decay")
    parser.add_argument(
        "--clip_grad_norm", type=float, default=1.0, help="clip grad norm"
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="save checkpoint every specified epochs",
    )  # 每隔多少轮保存一次模型
    parser.add_argument("--eval_freq", type=int, default=10, help="freqadv eval")  # 每隔多少轮进行评估

    # Data loading 数据加载相关参数
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

    # General model options 模型及特殊参数
    parser.add_argument("--model", type=str, default="akorn", help="model")
    parser.add_argument("--L", type=int, default=1, help="num of layers")
    parser.add_argument("--T", type=int, default=16, help="Timesteps")
    parser.add_argument("--ch", type=int, default=512, help="num of channels")
    parser.add_argument("--heads", type=int, default=8)

    # AKOrN options AKOrN 模型参数
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0, help="step size")
    parser.add_argument("--J", type=str, default="attn", help="connectivity")
    parser.add_argument("--use_omega", type=str2bool, default=True)
    parser.add_argument("--global_omg", type=str2bool, default=True)
    parser.add_argument("--learn_omg", type=str2bool, default=False)
    parser.add_argument("--init_omg", type=float, default=0.1)
    parser.add_argument("--nl", type=str2bool, default=True)

    parser.add_argument("--speed_test", action="store_true")  # 是否在训练过程中测量每次迭代的时间开销

    args = parser.parse_args()  # 根据解析的命令行参数初始化环境

    print("Exp name: ", args.exp_name)

    # 启用 torch.backends.cudnn 优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    
    # 设置随机种子确保可复现性
    if args.seed is not None:
        import random
        import numpy as np

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 用于控制多进程加载的 CPU 核分配
    def worker_init_fn(worker_id):
        os.sched_setaffinity(0, range(args.cpu_core_start, args.cpu_core_end))

    # 从指定路径加载数据集
    if args.data_root is not None:
        rootdir = args.data_root
    else:
        rootdir = "./data/sudoku"
    
    # 加载训练数据，设置批量大小和加载方式。
    trainloader = torch.utils.data.DataLoader(
        SudokuDataset(rootdir, train=True),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )
    # 加载测试数据，设置批量大小和加载方式。
    testloader = torch.utils.data.DataLoader(
        SudokuDataset(rootdir, train=False),
        batch_size=100,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    jobdir = f"runs/{args.exp_name}/"
    writer = SummaryWriter(jobdir)

    # only compute digit-wise accuracy
    from source.evals import compute_board_accuracy
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
            
            _, _, board_accuracy = compute_board_accuracy(out, Y, is_input)  # 每个批次中完全正确的棋盘数量
            # 累加当前批次的正确预测数量和总棋盘数量
            correct += board_accuracy.sum().item()
            total += board_accuracy.shape[0]
           
            # digit wise input accuracy
            out = out.argmax(dim=-1)
            Y = Y.argmax(dim=-1)
            mask = (1 - is_input).view(out.shape)
            correct_input += ((1 - mask) * (out == Y)).sum().item()
            total_input += (1 - mask).sum().item()

        acc = correct / total  # 棋盘总体准确率（即有每个棋盘完全正确的概率）
        input_acc = correct_input / total_input  # 输入区域的准确率（即平均每个棋盘输入位置正确的概率）
        return acc, input_acc, (total, correct), (total_input, correct_input)

    # 模型选择
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

    # 定义优化器、 EMA 更新器 和损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Exponential Moving Average（指数移动平均），用来平滑模型参数的变化， EMA 仅依赖 net 的权重，本身不参与反向传播从而提升模型的泛化性能
    ema = EMA(net, beta=args.beta, update_every=10, update_after_step=100)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # Measure speed
    if args.speed_test:
        it_sp = 0
        time_per_iter = []
    
    # 训练与评估循环
    import numpy as np 
    for epoch in range(args.epochs):
        total_loss = 0

        for X, Y, is_input in tqdm.tqdm(trainloader):
            net.train()
            ema.train()
            X, Y, is_input = X.to(torch.int32).cuda(), Y.cuda(), is_input.cuda()

            if args.speed_test:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            out = net(X, is_input)
            
            out = out.reshape(-1, 9)
            Y = Y.argmax(dim=-1).reshape(-1)
            
            loss = criterion(out, Y).mean()
            
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad_norm > 0.:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)
            optimizer.step()

            if args.speed_test:
                end.record()
                torch.cuda.synchronize()
                time_elapsed_per_iter = start.elapsed_time(end)
                time_per_iter.append(time_elapsed_per_iter)
                print(time_elapsed_per_iter)
                it_sp = it_sp + 1
                if it_sp == 100:
                    np.save(os.path.join(jobdir, "time.npy"), np.array(time_per_iter))
                    exit(0)

            total_loss += loss.item()
            ema.update()

        total_loss = total_loss / len(trainloader)

        writer.add_scalar("training loss", total_loss, epoch)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss:.4f}")

        if (epoch + 1) % args.eval_freq == 0:

            acc, input_acc, stats, stats_input = compute_acc(net, testloader)
            writer.add_scalar("test/accuracy", acc, epoch)
            writer.add_scalar("test/input_accuracy", input_acc, epoch)
            print(f"[Test]: Total blanks:{stats[0]}, Accuracy: {acc}")
            print(
                f"[Test]: Total given squares:{stats_input[0]}, Accuracy on given digits: {input_acc}"
            )

            # EMA evals
            acc, input_acc, stats, stats_input = compute_acc(ema.ema_model, testloader)
            writer.add_scalar("ema_test/accuracy", acc, epoch)
            writer.add_scalar("ema_test/input_accuracy", input_acc, epoch)
            print(f"[EMA Test]: Total blanks:{stats[0]}, Accuracy: {acc}")
            print(
                f"[EMA Test]: Total given squares:{stats_input[0]}, Accuracy on given digits: {input_acc}"
            )

        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(net, optimizer, epoch, total_loss, checkpoint_dir=jobdir)
            save_model(ema, epoch, checkpoint_dir=jobdir, prefix="ema")

    torch.save(net.state_dict(), os.path.join(jobdir, f"model.pth"))
    torch.save(ema.state_dict(), os.path.join(jobdir, f"ema_model.pth"))
