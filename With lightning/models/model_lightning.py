import pytorch_lightning as pl
from utils.PE import PatchEmbedding
from utils.Attention import MultiHeadAttention
from models.VIT import VIT
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from torch import argmax, tensor, stack
from data.dataload import generate_VITSet
from torch.utils.data import DataLoader
import argparse


# from numpy import mean

class VIT_lightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = VIT()
        self.loss = CrossEntropyLoss()
        self.n_test = None
        self.train_set, self.n_test = generate_VITSet(mode='train', valid_ratio=self.hparams.valid_ratio, n_test=self.n_test)
        self.val_set = generate_VITSet(mode='vali', valid_ratio=self.hparams.valid_ratio, n_test=self.n_test)

    def forward(self, data_x):
        return self.model(data_x)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), self.hparams.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {'optimizer':optimizer,'lr_scheduler':scheduler}

    def training_step(self, batch, batch_idx):
        data_x, data_y = batch

        outputs = self(data_x)
        loss = self.loss(outputs, data_y)
        self.log('training_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data_x, data_y = batch
        outputs = self(data_x)
        train_pred = argmax(outputs, dim=1)
        train_acc = train_pred == data_y
        loss = self.loss(outputs, data_y)
        acurate = sum(train_acc) / len(train_acc)
        # self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('val_acc', acurate, on_step=True, on_epoch=False, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acurate}

    # def validation_epoch_end(self, validation_step_outputs):
    #     val_out = [num for elem in validation_step_outputs for num in elem]
    #     acc = sum(val_out) / len(val_out)
    #     del val_out
    #     self.log('val_total_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VIT_lightning")
        parser.add_argument('-e', '--epoches', type=int, default=500,
                            help='input training epoch for training (default: 5)')
        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                            help='input learning rate for training (default: 5e-4)')
        return parent_parser
