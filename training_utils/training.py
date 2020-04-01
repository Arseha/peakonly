import os
import torch
import torch.nn as nn
import numpy as np


def accuracy(logits, y_true):
    """
    :param logits: np.ndarray, output of the model
    :param y_true: np.ndarray
    """
    predictions = np.argmax(logits, axis=1)
    correct_samples = np.sum(predictions == y_true)
    total_samples = y_true.shape[0]
    return float(correct_samples) / total_samples


def compute_accuracy(model, loader):
    """
    :param model: a model which returns classifier_output and segmentator_output
    :param loader: data loader
    """
    model.eval()  # enter evaluation mode
    score_accum = 0
    count = 0

    for x, y, _, _ in loader:
        classifier_output, _ = model(x)
        score_accum += accuracy(classifier_output.data.cpu().numpy(), y.data.cpu().numpy()) * y.shape[0]
        count += y.shape[0]

    return float(score_accum / count)


def iou(logits, y_true, smooth=1e-2):
    """
    :param logits: np.ndarray, output of the model
    :param y_true: np.ndarray
    :param smooth: float
    """
    batch_size, channels, samples = logits.shape
    values = np.zeros(channels)

    for i in range(channels):
        pred = logits[:, i, :] > 0.5
        gt = y_true[:, i, :].astype(np.bool)
        intersection = (pred & gt).sum(axis=1)
        union = (pred | gt).sum(axis=1)
        values[i] = np.mean((intersection + smooth) / (union + smooth))

    return np.mean(values)


def compute_iou(model, loader):
    """
    Computes intersection over union on the dataset wrapped in a loader
    :param model: a model which returns classifier_output and segmentator_output
    :param loader: data loader
    returns: IoU (jaccard index) for integration and intersection masks
    """
    model.eval()  # Evaluation mode
    integration_score = []
    intersection_score = []

    for x, _, integration_mask, intersection_mask in loader:
        _, segmentator_output = model(x)
        predicted_integration_mask = segmentator_output[:, 0, :].data.cpu().numpy()
        predicted_intersection_mask = segmentator_output[:, 1, :].data.cpu().numpy()

        integration_mask = integration_mask.data.cpu().numpy()
        intersection_mask = intersection_mask.data.cpu().numpy()

        integration_score.append(iou(predicted_integration_mask, integration_mask))
        intersection_score.append(iou(predicted_intersection_mask, intersection_mask))

    return np.mean(integration_score), np.mean(intersection_score)


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
        self.dice = DiceLoss()
        self.bce = WeightedBCE(weights)

    def __call__(self, output, target):
        return self.dice(output, target) + self.bce(output, target)


def train_model(model, loader, val_loader,
                optimizer, num_epoch,
                print_epoch=10,
                classification_metric=None,
                segmentation_metric=None,
                scheduler=None,
                label_criterion=None,
                integration_criterion=None,
                intersection_criterion=None,
                accumulation=1):
    loss_history = []
    train_classification_score_history = []
    train_segmentation_score_history = []
    val_classification_score_history = []
    val_segmentation_score_history = []
    best_score = 0
    for epoch in range(num_epoch):
        model.train()  # enter train mode
        loss_accum = 0
        classification_score_accum = 0
        segemntation_score_accum = 0
        count = 0
        step = 0
        for x, y, integration_mask, intersection_mask in loader:
            classifier_output, integrator_output = model(x)
            # classifier_output = classifier_output.view(1, -1)
            # calculate loss and gradients
            loss = torch.tensor(0, dtype=torch.float32, device=x.device)
            if label_criterion is not None:
                loss = loss + label_criterion(classifier_output, y)
            if integration_criterion is not None:
                loss = loss + integration_criterion(integrator_output[:, 0, :], integration_mask)
            if intersection_criterion is not None:
                loss = loss + intersection_criterion(integrator_output[:, 1, :], intersection_mask)
            loss.backward()

            step += 1
            if step == accumulation:  # accumulate loss over few batches
                optimizer.step()
                optimizer.zero_grad()
                step = 0

            if classification_metric is not None:
                classification_score_accum += classification_metric(classifier_output.detach().cpu().numpy(),
                                                                    y.detach().cpu().numpy()) * len(y)
            if segmentation_metric is not None:
                gt = np.stack((integration_mask.data.cpu().numpy(),
                               intersection_mask.data.cpu().numpy())).transpose(1, 0, 2)
                segemntation_score_accum += segmentation_metric(integrator_output.detach().cpu().sigmoid().numpy(),
                                                                gt) * len(y)
            loss_accum += loss
            count += len(y)
        loss_history.append(float(loss_accum / count))  # average loss over epoch
        train_classification_score_history.append(float(classification_score_accum / count))
        train_segmentation_score_history.append(float(segemntation_score_accum / count))

        model.eval()  # enter evaluation mode
        classification_score_accum = 0
        segemntation_score_accum = 0
        count = 0
        for x, y, integration_mask, intersection_mask in val_loader:
            classifier_output, integrator_output = model(x)
            if classification_metric is not None:
                classification_score_accum += classification_metric(classifier_output.detach().cpu().numpy(),
                                                                    y.detach().cpu().numpy()) * len(y)
            if segmentation_metric is not None:
                gt = np.stack((integration_mask.data.cpu().numpy(),
                               intersection_mask.data.cpu().numpy())).transpose(1, 0, 2)
                segemntation_score_accum += segmentation_metric(integrator_output.detach().cpu().sigmoid().numpy(),
                                                                gt) * len(y)
            count += len(y)
        val_classification_score_history.append(float(classification_score_accum / count))
        val_segmentation_score_history.append(float(segemntation_score_accum / count))

        # save best model based on classification score (if it is not None)
        if val_classification_score_history[-1] > 0:
            if best_score < val_classification_score_history[-1]:
                best_score = val_classification_score_history[-1]
                torch.save(model.state_dict(),
                           os.path.join('data/weights', model.__class__.__name__))  # save best model
        elif val_segmentation_score_history[-1] > 0:
            if best_score < val_segmentation_score_history[-1]:
                best_score = val_segmentation_score_history[-1]
                torch.save(model.state_dict(),
                           os.path.join('data/weights', model.__class__.__name__))  # save best model

        if scheduler:
            scheduler.step()

        if not epoch % print_epoch or epoch == num_epoch - 1:
            print('Epoch #{}, train loss: {:.4f}'.format(
                epoch, loss_history[-1]))
            if classification_metric is not None:
                print('Train classification score: {:.4f}, val classificiation score: {:.4f}'.format(
                    train_classification_score_history[-1],
                    val_classification_score_history[-1]
                ))
            if segmentation_metric is not None:
                print('Train segmentation score: {:.4f}, val segmentation score: {:.4f}'.format(
                    train_segmentation_score_history[-1],
                    val_segmentation_score_history[-1]
                ))
    return (loss_history,
            train_classification_score_history,
            train_segmentation_score_history,
            val_classification_score_history,
            val_segmentation_score_history)
