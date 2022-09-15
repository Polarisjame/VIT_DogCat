import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from tqdm import trange, tqdm
from torch import Tensor


def train(model, train_loader, valid_loader, epochs, lossF, optimizer, device, scheduler=None, saveround=5):
    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []
    for epoch in range(epochs):
        model.train()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='it') as pbar:
            for (data_x, data_y) in train_loader:
                train_epoch_loss = []
                data_x = data_x.to(device)
                data_y = data_y.to(device)
                outputs = model(data_x)
                optimizer.zero_grad()
                loss = lossF(outputs, data_y)
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())
                train_loss.append(loss.item())
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
        train_epochs_loss.append(np.average(train_epoch_loss))

        # =====================valid============================
        model.eval()
        valid_epoch_loss = []
        true_classes = []
        pre_classes = []
        correct = 0
        with tqdm(total=len(valid_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='it') as pbar:
            for (data_x, data_y) in valid_loader:
                data_x = data_x.to(device)
                data_y = data_y.to(device)
                true_class = data_y.to('cpu')
                true_classes.extend(true_class.numpy().tolist())
                outputs = model(data_x)
                loss = lossF(outputs, data_y)
                with torch.no_grad():
                    train_pred = torch.argmax(outputs, dim=1)
                    train_acc = (train_pred == data_y).float()
                    pre_class = train_pred.to('cpu')
                    correct += torch.sum(train_acc)
                    pre_classes.extend(pre_class.numpy().tolist())
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
        valid_epochs_loss.append(np.average(valid_epoch_loss))
        print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Recall: {:.0f}, F1score:{:.0f}\n'.format(
            np.mean(valid_epoch_loss), correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset), recall_score(true_classes, pre_classes),
            f1_score(true_classes, pre_classes)))
        # ====================adjust lr========================
        if scheduler is not None:
            scheduler.step()
        # lr_adjust = {
        #     2: 1e-4, 4: 5e-5, 6: 1e-5, 8: 5e-6,
        #     10: 1e-6, 15: 1e-7, 20: 5e-8
        # }
        # if epoch in lr_adjust.keys():
        #     lr = lr_adjust[epoch]
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        #     print('Updating learning rate to {}'.format(lr))
        if (epoch + 1) % saveround == 0:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, './models/model' + str(epoch + 1) + '.pth')
    return train_loss, valid_loss, train_epochs_loss, valid_epochs_loss
