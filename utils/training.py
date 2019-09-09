import os
import numpy as np
import torch
import torch.nn as nn


def train_model(model, loader, val_loader, criterion, metric, optimizer,
                num_epoch, device='cpu', scheduler=None):
    best_score = None
    loss_history = []
    val_loss_history = []
    val_score_history = []
    for epoch in range(num_epoch):
        model.train()  # enter train mode
        loss_accum = 0
        count = 0
        for x, y in loader:
            logits = model(x.to(device))
            loss = criterion(logits, y.to(device))
            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss accumulation
            count += 1
            loss_accum += loss
        loss_history.append(float(loss_accum / count))  # average loss over epoch

        model.eval()  # enter evaluation mode
        loss_accum = 0
        score_accum = 0
        count = 0
        for x, y in val_loader:
            logits = model(x.to(device))
            y = y.to(device)
            count += 1
            loss_accum += criterion(logits, y)
            score_accum += metric(logits, y)
        val_loss_history.append(float(loss_accum / count))
        val_score_history.append(float(score_accum / count))

        if best_score is None or best_score < np.mean(val_score_history[-1]):
            best_score = np.mean(val_score_history[-1])
            torch.save(model.state_dict(), os.path.join('data', model.__class__.__name__))  # save best model

        if scheduler:
            scheduler.step()  # make scheduler step

        print('Epoch #{}, train loss: {:.4f}, val loss: {:.4f}, {}: {:.4f}'.format(
            epoch,
            loss_history[-1],
            val_loss_history[-1],
            metric.__name__,
            val_score_history[-1]
        ))
    return loss_history, val_loss_history, val_score_history


def accuracy(logits, y_true):
    '''
    logits: torch.tensor on the device, output of the model
    y_true: torch.tensor on the device
    '''
    _, indices = torch.max(logits, 1)
    correct_samples = torch.sum(indices == y_true)
    total_samples = y_true.shape[0]
    return float(correct_samples) / total_samples


def compute_accuracy(model, loader, device):
    model.eval()
    score_accum = 0
    count = 0
    for x, y in loader:
        logits = model(x.to(device))
        count += 1
        score_accum += accuracy(logits, y.to(device))
    return float(score_accum / count)


def iou(logits, y_true, smooth=1e-2):
    batch_size, channels, samples = logits.shape
    values = torch.zeros(channels)
    for i in range(channels):
        pred = logits[:, i, :].sigmoid() > 0.5
        gt = y_true[:, i, :].bool()
        intersection = (pred & gt).float().sum(1)  # will be zero if Truth=0 or Prediction=0
        union = (pred | gt).float().sum(1)  # will be zero if both are 0
        values[i] = torch.mean((intersection + smooth) / (union + smooth))
    return torch.mean(values)


def compute_iou(model, loader, device):
    """
    Computes intersection over union on the dataset wrapped in a loader
    Returns: IoU (jaccard index)
    """
    model.eval()  # Evaluation mode
    IoUs_mask = []
    IoUs_domain = []
    for x, y in loader:
        mask = y[:, :1, :]
        domain = y[:, 1:, :]
        logits = model(x.to(device))
        predicted_mask = logits[:, :1, :]
        predicted_domain = logits[:, 1:, :]
        IoUs_mask.append(iou(predicted_mask, mask.to(device)).cpu().numpy())
        IoUs_domain.append(iou(predicted_domain, domain.to(device)).cpu().numpy())
    return np.mean(IoUs_mask), np.mean(IoUs_domain)


class WeightedBCE:
    def __init__(self, weights=None):
        self.weights = weights
        self.logsigmoid = nn.LogSigmoid()

    def __call__(self, output, target):
        if self.weights is not None:
            assert len(self.weights) == 2
            loss = self.weights[1] * (target * self.logsigmoid(output)) + \
                self.weights[0] * ((1 - target) * self.logsigmoid(-output))
        else:
            loss = target * self.logsigmoid(output) + (1 - target) * self.logsigmoid(-output)
        return torch.neg(torch.mean(loss))


class DiceLoss:
    def __init__(self, smooth=1e-2):
        self.smooth = smooth

    def __call__(self, output, target):
        output = output.sigmoid()
        numerator = torch.sum(output * target, dim=1)
        denominator = torch.sum(torch.sqrt(output) + target, dim=1)
        return 1 - torch.mean((2 * numerator + self.smooth) / (denominator + self.smooth))


class CombinedLoss:
    def __init__(self, weights=None):
        self.dice =DiceLoss()
        self.bce = WeightedBCE(weights)

    def __call__(self, output, target):
        return self.dice(output, target) + self.bce(output, target)


class TwoChannelLoss:
    def __init__(self, weights_split=None, weights_area=None):
        self.split_loss = CombinedLoss(weights_split)
        self.area_loss = CombinedLoss(weights_area)

    def __call__(self, output, target):
        return self.split_loss(output[:, 0, :], target[:, 0, :]) + \
               self.split_loss(output[:, 1, :], target[:, 1, :])
