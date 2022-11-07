import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from data.dataload import generate_VITSet



class VITSet():
    def __init__(self, **kwargs):
        super().__init__()
        # self.save_hyperparameters()
        self.valset = None
        self.trainset = None
        self.batch_size = kwargs['batch_size']
        self.valid_ratio = kwargs['valid_ratio']
        self.num_workers = kwargs['num_workers']
        self.n_test = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset, self.n_test = generate_VITSet(mode='train', valid_ratio=self.valid_ratio, n_test=self.n_test)
            self.valset = generate_VITSet(mode='vali', valid_ratio=self.valid_ratio, n_test=self.n_test)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=0)
