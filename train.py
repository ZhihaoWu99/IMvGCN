"""
@Project   : IMvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : train.py
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_evaluation_results, criteria
from Dataloader import load_data
from model import IMvGCN


def train(args, device):
    feature_list, flt_list, flt_f, labels, idx_labeled, idx_unlabeled = load_data(args, device=device)
    print('Labeled sample:', len(idx_labeled))
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    layers = [args.dim1, args.dim2]
    num_view = len(feature_list)
    input_dims = []
    for i in range(num_view):
        input_dims.append(feature_list[i].shape[1])

    model = IMvGCN(input_dims, num_classes, args.dropout, layers, device).to(device)
    loss_function1 = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    begin_time = time.time()

    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            model.train()
            output, hidden_list, w_list = model(feature_list, flt_list, flt_f)
            output = F.log_softmax(output, dim=1)
            optimizer.zero_grad()
            loss_nl = loss_function1(output[idx_labeled], labels[idx_labeled])
            loss_rl = criteria(num_view, output, w_list, feature_list, flt_list, args.Lambda)
            total_loss = loss_nl + args.alpha * loss_rl
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                output, _, _ = model(feature_list, flt_list, flt_f)
                pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                ACC, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
                pbar.set_postfix({'Loss': '{:.6f}'.format(total_loss.item()),
                                  'ACC': '{:.2f}'.format(ACC * 100),
                                  'F1': '{:.2f}'.format(F1 * 100)})
                pbar.update(1)

            del output, hidden_list, w_list
            torch.cuda.empty_cache()

    cost_time = time.time() - begin_time
    model.eval()
    output, _, _ = model(feature_list, flt_list, flt_f)
    print("Evaluating the model")
    pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
    ACC, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
    print("ACC: {:.2f}, F1: {:.2f}".format(ACC * 100, F1 * 100))

    return ACC, F1, cost_time
