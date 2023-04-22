"""
@Project   : IMvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : args.py
"""
import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="gpu number or cpu")
    parser.add_argument("--path", type=str, default="./data/datasets/", help="Dataset path")
    parser.add_argument("--dataset", type=str, default="Citeseer", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")

    parser.add_argument("--n_repeated", type=int, default=5, help="Repeated times")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--ratio", type=float, default=0.01, help="Label ratio")
    parser.add_argument("--num_epoch", type=int, default=300, help="Training epochs")

    parser.add_argument("--Lambda", type=float, default=0.5, help="Value of Lambda")
    parser.add_argument("--alpha", type=float, default=1e-5, help="Value of alpha")
    parser.add_argument("--dim1", type=int, default=8, help="hidden dimensions")
    parser.add_argument("--dim2", type=int, default=32, help="hidden dimensions")

    args = parser.parse_args()

    return args
