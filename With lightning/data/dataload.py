import torch.utils.data
from torchvision.datasets import ImageFolder
import random
from torchvision.transforms import Compose, Resize, ToTensor

# def load_data(batch_size, valid_ratio):
#
#     path = r'/CatDog/data'
#     data_train = ImageFolder(path, transform)
#     n = len(data_train)  # 数据集总数
#     n_test = random.sample(range(1, n), int(valid_ratio * n))  # 按比例取随机数列表
#     valid_set = torch.utils.data.Subset(data_train, n_test)  # 按照随机数列表取测试集
#     train_set = torch.utils.data.Subset(data_train, list(set(range(1, n)).difference(set(n_test))))  # 测试集剩下作为训练集
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
#     return train_loader, valid_loader, data_train

path = r'..\..\data\datas'
transform = Compose([Resize((224, 224)), ToTensor()])
global data_train
data_train = ImageFolder(path, transform)


def generate_VITSet(mode, valid_ratio, n_test=None):
    global data_train
    n = len(data_train)  # 数据集总数
    if n_test is None:
        n_test = random.sample(range(1, n), int(valid_ratio * n))
    if mode == 'train':
        train_set = torch.utils.data.Subset(data_train, list(set(range(1, n)).difference(set(n_test))))
        return train_set, n_test
    elif mode == 'vali':
        valid_set = torch.utils.data.Subset(data_train, n_test)
        return valid_set
