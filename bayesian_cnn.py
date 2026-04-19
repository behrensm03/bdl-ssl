import torch
import torch.nn as nn
from bayesian_layers import BayesianConv2d, BayesianLinear


class BayesianCNN(nn.Module):
    #initialize the model with these layers
    def __init__(self, in_channels, num_classes, prior_sigma=1.0):
        super().__init__()

        self.layer1 = nn.Sequential(
            BayesianConv2d(in_channels, 16, kernel_size=3, prior_sigma=prior_sigma),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            BayesianConv2d(16, 16, kernel_size=3, prior_sigma=prior_sigma),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            BayesianConv2d(16, 64, kernel_size=3, prior_sigma=prior_sigma),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            BayesianConv2d(64, 64, kernel_size=3, prior_sigma=prior_sigma),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            BayesianConv2d(64, 64, kernel_size=3, padding=1, prior_sigma=prior_sigma),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        #three linear layers, Bayesian
        self.fc1 = BayesianLinear(64 * 4 * 4, 128, prior_sigma=prior_sigma)
        self.fc2 = BayesianLinear(128, 128, prior_sigma=prior_sigma)
        self.fc3 = BayesianLinear(128, num_classes, prior_sigma=prior_sigma)

        self.relu = nn.ReLU()

    #forwards pass for the whole model
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        #flatten input with constant batch size (x.shape is (32,3,28,28))
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    #sum KL contribution from all layers 
    def kl_divergence(self):
        kl = torch.tensor(0.0, device=next(self.parameters()).device)

        for module in self.modules():
            if hasattr(module, "kl"):
                kl = kl + module.kl
        return kl