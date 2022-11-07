import torch
from pytorch_lightning.callbacks import ModelCheckpoint
# from torch.nn import CrossEntropyLoss
# from torch.optim import Adam, lr_scheduler

import numpy as np
import random
# from CatDog.models.VIT import VIT
import argparse
# from train import train
# from CatDog.data.dataload import load_data
from matplotlib import pyplot as plt
from data import VITSet
from torch.utils.data import random_split, DataLoader

from data.dataload import data_train
from models import VIT_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main(args):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=args.topk,
        mode='min',
        save_last=True,
        dirpath=r'E:\TOOL\pythonProgram\CatDog\dl_ckpt'
    )
    model = VIT_lightning(args)
    # n = len(data_train)
    # train, val = random_split(data_train, [int(0.9 * n), n - int(0.9 * n)])
    # data = VITSet(**vars(args))
    # data.setup()
    # train_set = data.train_dataloader()
    # val_set = data.val_dataloader()
    trainer = Trainer.from_argparse_args(args, max_epochs=args.epoches, accelerator='gpu', devices=1,
                                         auto_scale_batch_size=args.auto_batch, auto_lr_find=args.auto_lr, precision=16,
                                         callbacks=RichProgressBar(), limit_train_batches=100, limit_val_batches=10,
                                         )
    trainer.fit(model)
    # trainer.tune(model, datamodule=data)
    # print(model.hparams.batch_size, model.hparams.learning_rate)
    # trainer.fit(model,DataLoader(train, batch_size=8, shuffle=True, num_workers=1),DataLoader(val, batch_size=8, shuffle=False, num_workers=1))
    # print(model.hparams.batch_size,model.hparams.learning_rate)
    # device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    # # batch_size = 16;epoches=5;valid_ratio=0.1
    # model = VIT().to(device)
    # lossF = CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # train_loader, valid_loader, _ = load_data(args.batch_size, args.valid_ratio)
    # train_loss, valid_loss, train_epochs_loss, valid_epochs_loss = train(model, train_loader, valid_loader,
    #                                                                      args.epoches, lossF, optimizer, device,
    #                                                                      scheduler, args.saveround)
    #
    # fig1, ax = plt.subplots(12)
    # ax1 = ax[0, 0]
    # ax2 = ax[0, 1]
    # ax1.plot(train_loss)
    # ax2.plot(valid_loss)
    # fig2, ax = plt.subplots(12)
    # ax1 = ax[0, 0]
    # ax2 = ax[0, 1]
    # ax1.plot(train_epochs_loss)
    # ax2.plot(valid_epochs_loss)
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CatvsDog VIT")
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('-vr', '--valid_ratio', type=float, default=0.1, help='divide valid:train sets by train_ratio')
    parser.add_argument('-cuda', '--use_cuda', type=bool, default=True, help='Use cuda or not')
    parser.add_argument('-k', '--topk', type=int, default=5, help='save the topk model')
    parser.add_argument('-nu', '--num_workers', type=int, default=0, help='thread number')
    parser.add_argument('--auto_batch', type=bool, default=False, help='auto select batch size')
    parser.add_argument('--auto_lr', type=bool, default=False, help='auto select learnign rate')
    parser = VIT_lightning.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
