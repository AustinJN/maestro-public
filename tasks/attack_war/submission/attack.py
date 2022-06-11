from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(self, vm, device, attack_path, epsilon = 0.2, min_val = 0, max_val = 1):
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.epsilon = 0.07
        self.alpha = 2/335
        self.steps = 7
        self.random_start = False
        self._supported_mode = ['default', 'targeted']
        self.min_val = 0
        self.max_val = 1

    def attack(
        self, original_images: np.ndarray, labels: List[int], target_label = None):
        original_images = original_images.to(self.device)
        labels = torch.tensor(labels).to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        perturbed_image = original_images

        loss = nn.CrossEntropyLoss()

        if self.random_start:
            # Starting at a uniformly random point
            perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-self.epsilon, self.epsilon)
            perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val).detach()

        for _ in range(self.steps):
            data_grad = self.vm.get_batch_input_gradient(perturbed_image, labels, loss)
            perturbed_image = perturbed_image.detach() - self.alpha * data_grad.sign()
            delta = torch.clamp(perturbed_image - original_images, min = -self.epsilon, max = self.epsilon)
            perturbed_image = torch.clamp(original_images + delta, min = 0, max = 1).detach()

        adv_outputs, detected_output  = self.vm.get_batch_output(perturbed_image)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        return perturbed_image.cpu().detach().numpy(), correct

