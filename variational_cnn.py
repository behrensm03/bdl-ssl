# This will contain the VI version of our BCNN

# Goal 1: start with a simple variational layer and verify it works
# Goal 2: extend to a convolutional layer and verify it works
# Goal 3: compose multiple layers into a full model

import numpy as np
import torch
import torch.nn as nn

class VariationalLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, prior_mean=0.0, prior_var=1.0):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.prior_mean = prior_mean
        self.prior_var = prior_var

        # Initialize variatonal parameters for our layer's weights        
        self.mu_w = nn.Parameter(torch.zeros(out_features, in_features)) # mean values, initialized randomly
        nn.init.kaiming_normal_(self.mu_w)
        self.mu_bias = nn.Parameter(torch.zeros(out_features))

        rho_init = -2.25 # this makes it so that the initial sigma is around 0.1, which is a common choice for initialization
        self.r_w = nn.Parameter(torch.full((out_features, in_features), rho_init)) # unconstrained standard deviation initialized randomly, later we will need to call softplus to get sigma
        self.r_bias = nn.Parameter(torch.full((out_features,), rho_init))
        return

    def forward(self, x):
        # This samples a new set of weights and biases from the current variational distribution each time we call forward
        # So to run S network samples on an input batch, we just call forward S times
        # Importantly, two different batches will use different samples of weights.
        sigma_w = torch.nn.functional.softplus(self.r_w)
        sigma_bias = torch.nn.functional.softplus(self.r_bias)

        epsilon_w = torch.randn_like(self.mu_w)
        epsilon_bias = torch.randn_like(self.mu_bias)

        w = self.mu_w + sigma_w * epsilon_w
        bias = self.mu_bias + sigma_bias * epsilon_bias

        return torch.nn.functional.linear(x, w, bias)

    def kl_divergence(self):
        # for 2 gaussians this has closed form solution
        # see kingma and welling appendix B for formula:
        # -KL(q || p) = 0.5 * sum [ 1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 ]

        sigma_w = torch.nn.functional.softplus(self.r_w)
        sigma_bias = torch.nn.functional.softplus(self.r_bias)

        kl_w = 0.5 * torch.sum(1 + torch.log(sigma_w**2) - self.mu_w**2 - sigma_w**2)
        kl_bias = 0.5 * torch.sum(1 + torch.log(sigma_bias**2) - self.mu_bias**2 - sigma_bias**2)

        return (kl_w + kl_bias) * -1.0


class VariationalConv2DLayer(nn.Module):
    # shoud follow similar logic to above but use Conv2d instead of linear
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, prior_mean=0.0, prior_var=1.0):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def kl_divergence(self):
        pass
    
class VariationalCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VariationalCNN, self).__init__()
        # implement layers here
        pass
        
    def forward(self, x):
        # sample a new set of weights and run through each layer
        pass
    
    def kl_divergence(self):
        # accumulate the kl divergence of each layer
        pass
    
    def average_probs(self, x, num_samples=10):
        # Run the input through the network S times, each time sampling a new set of weights
        # Then, compute the average of the S output prediction probabilities to get an average vector
        pass
