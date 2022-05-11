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
        min_val = 0,
        max_val = 1
    ):
        """
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            min_val: minimum value of each element in original image
            max_val: maximum value of each element in original image
                     each element in perturbed image should be in the range of min_val and max_val
            attack_path: Any other sources files that you may want to use like models should be available in ./submissions/ folder and loaded by attack.py. 
                         Server doesn't load any external files. Do submit those files along with attack.py
        """
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
        original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        labels = target_labels if target_labels != None else labels
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

        adv_outputs = self.vm.get_batch_output(perturbed_image)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        return np.squeeze(perturbed_image.cpu().detach().numpy()), correct