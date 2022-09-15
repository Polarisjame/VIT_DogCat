import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler

import numpy as np
import random
from VIT import VIT
import argparse
from train import train
from dataload import load_data
from matplotlib import pyplot as plt

seed = 42
torch.manual_seed(seed)
model = VIT()
np.random.seed(seed)
random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="CatvsDog VIT")
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('-e', '--epoches', type=int, default='5', help='input training epoch for training (default: 5)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4,
                        help='input learning rate for training (default: 5e-4)')
    parser.add_argument('-vr', '--valid_ratio', type=float, default=0.1, help='divide valid:train sets by train_ratio')
    parser.add_argument('-cuda', '--use_cuda', type=bool, default=True, help='Use cuda or not')
    parser.add_argument('-s', '--saveround', type=int, default=5, help='save the model per n(default:5) round of train')
    args = parser.parse_args()
    ops = vars(args)  # 将namespace类型转化为字典类型
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    # batch_size = 16;epoches=5;valid_ratio=0.1
    model = VIT().to(device)
    lossF = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    train_loader, valid_loader, _ = load_data(args.batch_size, args.valid_ratio)
    train_loss, valid_loss, train_epochs_loss, valid_epochs_loss = train(model, train_loader, valid_loader,
                                                                         args.epoches, lossF, optimizer, device,
                                                                         scheduler, args.saveround)

    fig1, ax = plt.subplots(12)
    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax1.plot(train_loss)
    ax2.plot(valid_loss)
    fig2, ax = plt.subplots(12)
    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax1.plot(train_epochs_loss)
    ax2.plot(valid_epochs_loss)
    plt.show()


if __name__ == "__main__":
    main()
