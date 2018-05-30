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
        #return self.add_noise(bin_samples)
        return bin_samples.float()

    def add_noise(self, attributes):
        noise = self.noise.sample((attributes.shape[0],))
        return attributes.float() + noise

ATTRIBUTES = ['Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Male',
              'No_Beard', 'Smiling', 'Wearing_Hat', 'Young']

def generate_fixed(generator, all_attributes):
    full_lst = []
    fixed_attributes = generator.sample(8)
    for i, attr in enumerate(ATTRIBUTES):
        idx = all_attributes.index(attr)
        fixed = fixed_attributes[i]
        for x in [-1, -0.75, -0.5, -0.2, 0.2, 0.5, 0.75, 1]:
            new = fixed.clone()
            new[idx] = x
            full_lst.append(new)
    return torch.stack(full_lst, 0)

def mismatch_attributes(attributes):
    # Inverts the attributes
    return -attributes

def smooth_labels(labels, device, type, strength=0.2):
    if strength == 0:
        print('zero')
        return labels
    # Labels needs to be in {0,1}
    N = labels.shape
    res = torch.zeros(N, device=device)
    res += labels * (1 - strength * torch.rand(N).type(type))
    #res += (1-labels) * strength * torch.rand(N).type(type)  # Comment for one-sided
    return res

def add_noise(imgs, start_strength, end_epoch, current_epoch, device=None):
    if device is not None:
        return imgs + max(0, start_strength - start_strength * current_epoch / (
            end_epoch)) * torch.randn(size=imgs.shape, device=device)
    return imgs + max(0,start_strength - start_strength * current_epoch / (
        end_epoch)) * torch.randn(size=imgs.shape)


def flip_labels(labels, prob, type):
    labels += (1-2*labels)*torch.bernoulli(torch.full(labels.shape, prob)).type(type)
    return labels
