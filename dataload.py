import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader,Subset
import random




def load_data(batch_size, valid_ratio):
    transform = Compose([Resize((224, 224)), ToTensor()])
    path = r'E:\TOOL\pythonProgram\CatDog\data'
    data_train = ImageFolder(path, transform)
    n = len(data_train)  # 数据集总数
    n_test = random.sample(range(1, n), int(valid_ratio * n))  # 按比例取随机数列表
    valid_set = torch.utils.data.Subset(data_train, n_test)  # 按照随机数列表取测试集
    train_set = torch.utils.data.Subset(data_train, list(set(range(1, n)).difference(set(n_test))))  # 测试集剩下作为训练集
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader,data_train
