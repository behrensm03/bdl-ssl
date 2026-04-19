import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus(x):
    return torch.log1p(torch.exp(x))


def gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # computes the log-probaibllity of each weight under a gaussian
    # returns a tensor of the values
    return -0.5 * (
        math.log(2 * math.pi)
        + 2 * torch.log(sigma)
        + ((x - mu) ** 2) / (sigma ** 2)
    )

#linear layer class inhertiing from nn.Module
class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #stdev from gaussian prior
        self.prior_sigma = prior_sigma

        # mean of the weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.05))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).fill_(-5.0))

        # mean of the biases
        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0, 0.05))
        self.bias_rho = nn.Parameter(torch.empty(out_features).fill_(-5.0))

        #KL term from the layer is stored
        self.kl = torch.tensor(0.0)

    def forward(self, x):
        #apply softplus on rho to get sigma
        weight_sigma = softplus(self.weight_rho)
        bias_sigma = softplus(self.bias_rho)

        #create tensor of values sampled from N(0,1) in shape of weight_mu and 
        #bias_mu
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        #Use reparameterization trick
        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        #compute log probability with sampled values from q(theta)
        log_qw = gaussian_log_prob(weight, self.weight_mu, weight_sigma).sum()
        log_qb = gaussian_log_prob(bias, self.bias_mu, bias_sigma).sum()

        #create tensor filled with zeroes with same shape as weight and bias
        #also create tensors with same shape filled with the prior_sigma
        prior_mu_w = torch.zeros_like(weight)
        prior_sigma_w = torch.full_like(weight, self.prior_sigma)
        prior_mu_b = torch.zeros_like(bias)
        prior_sigma_b = torch.full_like(bias, self.prior_sigma)

        #calculate logg probability of sampled weights under prior
        log_pw = gaussian_log_prob(weight, prior_mu_w, prior_sigma_w).sum()
        log_pb = gaussian_log_prob(bias, prior_mu_b, prior_sigma_b).sum()

        # compute the KL for the layers sampled weights
        self.kl = (log_qw + log_qb) - (log_pw + log_pb)

        #aply linear layer multiplying input by weights and adding biases
        return F.linear(x, weight, bias)


#convolutional layer class 
class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, prior_sigma=1.0):
        super().__init__()
        self.in_channels = in_channels #number of input channels 3 for rgb
        self.out_channels = out_channels #number of filters
        #size of the kernel as int or tuple
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self.stride = stride
        self.padding = padding

        #stdev of Gaussian prior over weights
        self.prior_sigma = prior_sigma

        #shape for convolutional filters
        weight_shape = (out_channels, in_channels, *self.kernel_size)
        
        #variationl parameter for convolutional weights
        self.weight_mu = nn.Parameter(torch.empty(weight_shape).normal_(0, 0.05))
        self.weight_rho = nn.Parameter(torch.empty(weight_shape).fill_(-5.0))

        #variational parameters for the biases
        self.bias_mu = nn.Parameter(torch.empty(out_channels).normal_(0, 0.05))
        self.bias_rho = nn.Parameter(torch.empty(out_channels).fill_(-5.0))

        #KL contribution from this layer
        self.kl = torch.tensor(0.0)

    #single forward pass for convolution layer
    def forward(self, x):
        # use the softplus to compute the stdev
        weight_sigma = softplus(self.weight_rho)
        bias_sigma = softplus(self.bias_rho)

        #in shape of weight_mu and bias_mu create tensors sampled from N(0,1)
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        # use reparameterization trick
        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        #compute gaussian log probability of sampled weights under variation
        #distrubiton q
        log_qw = gaussian_log_prob(weight, self.weight_mu, weight_sigma).sum()
        log_qb = gaussian_log_prob(bias, self.bias_mu, bias_sigma).sum()

        #create tensor filled with zeroes with same shape as weight and bias
        #also create tensors with same shape filled with the prior_sigma
        prior_mu_w = torch.zeros_like(weight)
        prior_sigma_w = torch.full_like(weight, self.prior_sigma)
        prior_mu_b = torch.zeros_like(bias)
        prior_sigma_b = torch.full_like(bias, self.prior_sigma)

        #calculate logg probability of sampled weights under prior
        log_pw = gaussian_log_prob(weight, prior_mu_w, prior_sigma_w).sum()
        log_pb = gaussian_log_prob(bias, prior_mu_b, prior_sigma_b).sum()

        # compute the KL for the layers sampled weights
        self.kl = (log_qw + log_qb) - (log_pw + log_pb)

        #apply convolutional layer on input x using sampled filters and biases
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)