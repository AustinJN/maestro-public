"""
from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        epsilon = 0.2,
        min_val = 0,
        max_val = 1
    ):
        ###
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            epsilon: magnitude of perturbation that is added
        ###
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        #------------------------#
        self.epsilon = 0.07
        self.alpha = 2/335
        self.steps = 7
        self.random_start = False
        self._supported_mode = ['default', 'targeted']
        #------------------------#
        self.min_val = 0
        self.max_val = 1

    def attack(
        self, original_images: np.ndarray, labels: List[int], target_label = None,
    ):
        original_images = original_images.to(self.device)
        # original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        perturbed_image = original_images

        # -------------------- TODO ------------------ #

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

        # ------------------ END TODO ---------------- #

        adv_outputs, detected_output  = self.vm.get_batch_output(perturbed_image)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        return perturbed_image.cpu().detach().numpy(), correct
"""

from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Attack:
    # PGD Attack
    def __init__(self, vm, device, attack_path, epsilon=0.2, alpha=0.1, min_val=0, max_val=1, max_iters=10,  _type='linf'):
        # self.model = model._to(device)
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha
        self.max_iters = max_iters
        self._type = _type

    def project(self, x, original_x, epsilon, _type='linf'):
        if _type == 'linf':
            max_x = original_x + epsilon
            min_x = original_x - epsilon
            x = torch.max(torch.min(x, max_x), min_x)
        elif _type == 'l2':
            dist = (x - original_x)
            dist = dist.view(x.shape[0], -1)
            dist_norm = torch.norm(dist, dim=1, keepdim=True)
            mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
            # dist = F.normalize(dist, p=2, dim=1)
            dist = dist / dist_norm
            dist *= epsilon
            dist = dist.view(x.shape)
            x = (original_x + dist) * mask.float() + x * (1 - mask.float())
        else:
            raise NotImplementedError
        return x

    def attack(self, original_images, labels, target_label = None, reduction4loss='mean', random_start=False):
        # original_images = torch.unsqueeze(original_images, 0).to(self.device)
        original_images = original_images.to(self.device)
        # print(original_images.shape)
        # exit()
        labels = torch.tensor(labels).to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)

        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.to(self.device)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        # x.requires_grad = True

        # max_x = original_images + self.epsilon
        # min_x = original_images - self.epsilon

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                grads = self.vm.get_batch_input_gradient(x.data, target_labels)
                x.data -= self.alpha * torch.sign(grads.data)
                x = self.project(x, original_images, self.epsilon, self._type)
                x.clamp_(self.min_val, self.max_val)
                outputs = self.vm.get_batch_output(x)

        final_pred = outputs.max(1, keepdim=True)[1]

        correct = 0
        correct += (final_pred == target_labels).sum().item()
        # if final_pred.item() != labels.item():
        #     correct = 1
        # # return x
        return x.cpu().detach().numpy(), correct

