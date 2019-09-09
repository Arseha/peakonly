import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.models import Classifier
from utils.dataset import ROIDataset, get_label
from utils.training import train_model, accuracy, compute_accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is {}'.format(device))

    batch_size = 128

    model = Classifier().to(device)

    train_dataset = ROIDataset(path='data/train', key=get_label, mode='classification', gen_p=0.7)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = ROIDataset(path='data/val', key=get_label, mode='classification', gen_p=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    result = train_model(model, train_loader, val_loader, criterion,
                         accuracy, optimizer, 200, device, scheduler)

    model.load_state_dict(torch.load(os.path.join('data', model.__class__.__name__)))
    test_dataset = ROIDataset(path='data/test', key=get_label, mode='classification', gen_p=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print('test_accuracy: {:.4f}'.format(compute_accuracy(model, test_loader, device)))
