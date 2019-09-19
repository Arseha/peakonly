import torch
import torch.nn as nn

# change smth
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, padding=2, dilation=1, stride=1):
        super().__init__()

        if bn:
            self.basic_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 5, padding=padding, dilation=dilation, stride=stride),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        else:
            self.basic_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 5, padding=padding, dilation=dilation, stride=stride),
                nn.ReLU()
            )

    def forward(self, x):
        return self.basic_block(x)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.convBlock = nn.Sequential(
            Block(1, 8, False),
            nn.MaxPool1d(kernel_size=2),
            Block(8, 16, False),
            nn.MaxPool1d(kernel_size=2),
            Block(16, 32, False),
            nn.MaxPool1d(kernel_size=2),
            Block(32, 64, False),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 64, False),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 128, False),
            nn.MaxPool1d(kernel_size=2),
            Block(128, 256, False),
            nn.MaxPool1d(kernel_size=2),
            Block(256, 512, False)
        )
        self.classification = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.convBlock(x)
        x, _ = torch.max(x, dim=2)
        return self.classification(x)


def preprocessing(batch):
    batch_size, _, n_peaks = batch.shape
    processed_batch = batch.clone().view(batch_size, n_peaks)
    # TO DO: rewrite without loop
    for x in processed_batch:
        x[x < 1e-4] = 0
        pos = (x != 0)
        x[pos] = torch.log10(x[pos])
        x[pos] = x[pos] - torch.min(x[pos])
        x[pos] = x[pos] / torch.max(x[pos])
    return processed_batch.view(batch_size, 1, n_peaks)


class Integrator(nn.Module):
    def __init__(self, length=256):
        super().__init__()

        self.starter = nn.Sequential(
            Block(2, 16),
            Block(16, 20),
            nn.AvgPool1d(kernel_size=2),
            Block(20, 24),
            nn.AvgPool1d(kernel_size=2),
            Block(24, 28),
            nn.AvgPool1d(kernel_size=2),
            Block(28, 32)
        )

        self.pass_down1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            Block(32, 48)
        )

        self.pass_down2 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            Block(48, 64)
        )

        self.code = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            Block(64, 96),
            nn.Upsample(scale_factor=2),
            Block(96, 64)
        )

        self.pass_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Block(128, 64)
        )

        self.pass_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Block(112, 48)
        )

        self.finisher = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Block(80, 64),
            nn.Upsample(scale_factor=2),
            Block(64, 32),
            nn.Upsample(scale_factor=2),
            Block(32, 16),
            nn.Conv1d(16, 2, 1, padding=0)
        )

    def forward(self, x):
        x = torch.cat((x, preprocessing(x)), dim=1)
        starter = self.starter(x)
        pass1 = self.pass_down1(starter)
        pass2 = self.pass_down2(pass1)
        x = self.code(pass2)
        x = torch.cat((x, pass2), dim=1)
        x = self.pass_up2(x)
        x = torch.cat((x, pass1), dim=1)
        x = self.pass_up1(x)
        x = torch.cat((x, starter), dim=1)
        x = self.finisher(x)
        return x

