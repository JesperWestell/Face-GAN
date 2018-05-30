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
    def __init__(self, attributes, eps=0.25):
        self.attributes = attributes

    def sample(self, num_samples):
        idx = np.random.randint(0, len(self.attributes),num_samples)
        return torch.Tensor(self.attributes[idx,:])

    def sample_fixed(self, fixed_idx):
        attributes = self.attributes[self.attributes[:,fixed_idx] == 1,:]
        idx = np.random.randint(0, len(attributes))
        return torch.Tensor(attributes[idx, :])

ATTRIBUTES = ['Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Male',
              'No_Beard', 'Smiling', 'Wearing_Hat', 'Young']

def generate_fixed(generator, all_attributes):
    full_lst = []
    for i, attr in enumerate(ATTRIBUTES):
        idx = all_attributes.index(attr)
        fixed = generator.sample_fixed(idx)
        for x in [-1, -0.75, -0.5, -0.2, 0.2, 0.5, 0.75, 1]:
            new = fixed.clone()
            new[idx] = x
            full_lst.append(new)
    return torch.stack(full_lst, 0)

def mismatch_attributes(attributes, mismatch_prob = 0.25):
    # Inverts the attributes with a probability of mismatch_prob
    idx = torch.ones(attributes.shape).type(attributes.type())
    idx[torch.rand(attributes.shape) < mismatch_prob] = -1
    return attributes*idx

def smooth_labels(labels, device, type, strength=0.2):
    if strength == 0:
        return labels
    # Labels needs to be in {0,1}
    N = labels.shape
    res = torch.zeros(N, device=device)
    res += labels * (1 - strength * torch.rand(N).type(type))
    #res += (1-labels) * strength * torch.rand(N).type(type)  # Comment for one-sided
    return res

def add_noise(imgs, start_strength, end_epoch, current_epoch, device=None):
    if current_epoch >= end_epoch:
        return imgs
    if device is not None:
        return imgs + max(0, start_strength - start_strength * current_epoch / (
            end_epoch)) * torch.randn(size=imgs.shape, device=device)
    return imgs + max(0,start_strength - start_strength * current_epoch / (
        end_epoch)) * torch.randn(size=imgs.shape)


def flip_labels(labels, prob, type):
    labels += (1-2*labels)*torch.bernoulli(torch.full(labels.shape, prob)).type(type)
    return labels
