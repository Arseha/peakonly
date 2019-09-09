import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.models import Integrator
from utils.dataset import ROIDataset, get_mask, Reflection
from utils.training import train_model, iou, compute_iou, TwoChannelLoss

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is {}'.format(device))

    batch_size = 128

    model = Integrator().to(device)

    train_dataset = ROIDataset(path='data/train', augmentation=[Reflection(p=0.5)],
                               key=get_mask, mode='integration', gen_p=0.7)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = ROIDataset(path='data/val', key=get_mask, mode='integration', gen_p=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = optim.Adam(params=model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    criterion = TwoChannelLoss(weights_split=[0.1, 2], weights_area=[0.4, 0.2])

    result = train_model(model, train_loader, val_loader, criterion,
                         iou, optimizer, 150, device, scheduler)

    model.load_state_dict(torch.load(os.path.join('data', model.__class__.__name__), map_location=device))
    test_dataset = ROIDataset(path='data/test', key=get_mask, mode='integration', gen_p=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    IoU_splitter, IoU_integration = compute_iou(model, test_loader, device)
    print('test IoUs: {:.4f}, {:.4f}'.format(IoU_splitter, IoU_integration))
