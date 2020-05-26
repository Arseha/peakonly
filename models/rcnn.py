import torch
import torch.nn as nn


class EncodingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoding = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoding(x).transpose(2, 1)


class RecurrentCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoding = EncodingCNN()
        self.biLSTM = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.LSTM = nn.LSTM(128, 128, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(128, 2)
        self.integrator = nn.Linear(128, 2)

    def _preprocessing(self, batch):
        batch_size, _, n_points = batch.shape
        processed_batch = batch.clone().view(batch_size, n_points)
        # TO DO: rewrite without loop
        for x in processed_batch:
            x[x < 1e-4] = 0
            pos = (x != 0)
            x[pos] = torch.log10(x[pos])
            x[pos] = x[pos] - torch.min(x[pos])
            x[pos] = x[pos] / torch.max(x[pos])
        return processed_batch.view(batch_size, 1, n_points)

    def forward(self, x):
        x = torch.cat((x, self._preprocessing(x)), dim=1)
        x = self.encoding(x)
        x, _ = self.biLSTM(x)
        integrator_input, (classifier_input, _) = self.LSTM(x)
        classifier_output = self.classifier(classifier_input[0])
        integrator_output = self.integrator(integrator_input)
        return classifier_output, integrator_output.transpose(2, 1)
