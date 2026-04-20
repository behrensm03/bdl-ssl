import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, prior_mean=0.0, prior_var=1.0, rho_init=-2.25):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.prior_mean = prior_mean
        self.prior_var = prior_var

        # Initialize variatonal parameters for our layer's weights        
        self.mu_w = nn.Parameter(torch.zeros(out_features, in_features)) # mean values, initialized randomly
        nn.init.kaiming_normal_(self.mu_w)
        # nn.init.normal_(self.mu_w, mean=0.0, std=0.01)
        self.mu_bias = nn.Parameter(torch.zeros(out_features))

        # rho_init = -2.25 # this makes it so that the initial sigma is around 0.1, which is a common choice for initialization
        # print("rho_init lin:", rho_init)
        self.r_w = nn.Parameter(torch.full((out_features, in_features), rho_init)) # unconstrained standard deviation initialized randomly, later we will need to call softplus to get sigma
        self.r_bias = nn.Parameter(torch.full((out_features,), rho_init))
        return

    def forward(self, x):
        # This samples a new set of weights and biases from the current variational distribution each time we call forward
        # So to run S network samples on an input batch, we just call forward S times
        # Importantly, two different batches will use different samples of weights.
        sigma_w = F.softplus(self.r_w)
        sigma_bias = F.softplus(self.r_bias)

        epsilon_w = torch.randn_like(self.mu_w)
        epsilon_bias = torch.randn_like(self.mu_bias)

        w = self.mu_w + sigma_w * epsilon_w
        bias = self.mu_bias + sigma_bias * epsilon_bias

        return F.linear(x, w, bias)

    def kl_divergence(self):
        # for 2 gaussians this has closed form solution
        # see kingma and welling appendix B for formula:
        # -KL(q || p) = 0.5 * sum [ 1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 ]

        sigma_w = F.softplus(self.r_w)
        sigma_bias = F.softplus(self.r_bias)

        kl_w = 0.5 * torch.sum(1 + torch.log(sigma_w**2) - self.mu_w**2 - sigma_w**2)
        kl_bias = 0.5 * torch.sum(1 + torch.log(sigma_bias**2) - self.mu_bias**2 - sigma_bias**2)

        return (kl_w + kl_bias) * -1.0


class VariationalConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, prior_mean=0.0, prior_var=1.0, rho_init=-2.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.prior_mean = prior_mean
        self.prior_var = prior_var

        # Initialize variatonal parameters for our layer's weights        
        self.mu_w = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size)) # mean values, initialized randomly
        nn.init.kaiming_normal_(self.mu_w)
        # nn.init.normal_(self.mu_w, mean=0.0, std=0.01)
        self.mu_bias = nn.Parameter(torch.zeros(out_channels))

        # rho_init = -2.25 # this makes it so that the initial sigma is around 0.1, which is a common choice for initialization
        # print("rho_init conv:", rho_init)
        self.r_w = nn.Parameter(torch.full((out_channels, in_channels, kernel_size, kernel_size), rho_init)) # unconstrained standard deviation initialized randomly, later we will need to call softplus to get sigma
        self.r_bias = nn.Parameter(torch.full((out_channels,), rho_init))
        return

    def forward(self, x):
        sigma_w = F.softplus(self.r_w)
        sigma_bias = F.softplus(self.r_bias)

        epsilon_w = torch.randn_like(self.mu_w)
        epsilon_bias = torch.randn_like(self.mu_bias)

        w = self.mu_w + sigma_w * epsilon_w
        bias = self.mu_bias + sigma_bias * epsilon_bias

        return F.conv2d(x, w, bias, stride=self.stride, padding=self.padding)

    def kl_divergence(self):
        # for 2 gaussians this has closed form solution
        # see kingma and welling appendix B for formula:
        # -KL(q || p) = 0.5 * sum [ 1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 ]

        sigma_w = F.softplus(self.r_w)
        sigma_bias = F.softplus(self.r_bias)

        kl_w = 0.5 * torch.sum(1 + torch.log(sigma_w**2) - self.mu_w**2 - sigma_w**2)
        kl_bias = 0.5 * torch.sum(1 + torch.log(sigma_bias**2) - self.mu_bias**2 - sigma_bias**2)

        return (kl_w + kl_bias) * -1.0
    
class VariationalCNN(nn.Module):
    def __init__(self, in_channels, num_classes, prior_mean=0.0, prior_var=1.0, rho_init=-2.25):
        super(VariationalCNN, self).__init__()

        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            VariationalConv2DLayer(in_channels, 16, 3, prior_mean=prior_mean, prior_var=prior_var, rho_init=rho_init),
            nn.BatchNorm2d(16),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            VariationalConv2DLayer(16, 16, 3, prior_mean=prior_mean, prior_var=prior_var, rho_init=rho_init),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            VariationalConv2DLayer(16, 64, 3, prior_mean=prior_mean, prior_var=prior_var, rho_init=rho_init),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            VariationalConv2DLayer(64, 64, 3, prior_mean=prior_mean, prior_var=prior_var, rho_init=rho_init),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer5 = nn.Sequential(
            VariationalConv2DLayer(64, 64, 3, padding=1, prior_mean=prior_mean, prior_var=prior_var, rho_init=rho_init),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Sequential(
            VariationalLinearLayer(64 * 4 * 4, 128, prior_mean=prior_mean, prior_var=prior_var, rho_init=rho_init),
            nn.ReLU(),
            VariationalLinearLayer(128, 128, prior_mean=prior_mean, prior_var=prior_var, rho_init=rho_init),
            nn.ReLU(),
            VariationalLinearLayer(128, num_classes, prior_mean=prior_mean, prior_var=prior_var, rho_init=rho_init))
        
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.fc]
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    # TODO: may need to scale this so it doesn't dominate the loss
    def kl_divergence(self):
        kl = 0
        for layer in self.layers:
            for module in layer:
                if isinstance(module, VariationalConv2DLayer) or isinstance(module, VariationalLinearLayer):
                    kl += module.kl_divergence()
        return kl
    
    def average_probs(self, x, num_samples=10):
        # Run the input through the network S times, each time sampling a new set of weights
        # Then, compute the average of the S output prediction probabilities to get an average vector
        predictions = torch.zeros(num_samples, x.size(0), self.num_classes).to(x.device)
        for i in range(num_samples):
            predictions[i] = torch.nn.functional.softmax(self.forward(x), dim=1)

        return torch.mean(predictions, dim=0)
