import torch
import torch.nn as nn
import numpy as np


class PrintLayer(nn.Module):
    # For Debugging purposes
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, input):
        print(input.size())
        return input

class AttributeGenerator():
    # Generates attributes given a binomial distribution from data
    # Assumes all attributes are independent, which is most probably not
    # completely true
    def __init__(self, attributes, eps=0.25):
        self.probs = torch.from_numpy((np.mean((attributes+1)/2, axis=0)))
        self.binomial = torch.distributions.Binomial(probs=self.probs)
        self.noise = torch.distributions.normal.Normal(
            torch.zeros(attributes.shape[1]),
            torch.full((attributes.shape[1],), eps)
        )

    def sample(self, num_samples):
        bin_samples = 2*self.binomial.sample((num_samples,))-1
        return self.add_noise(bin_samples)

    def add_noise(self, attributes):
        noise = self.noise.sample((attributes.shape[0],))
        return attributes.float() + noise