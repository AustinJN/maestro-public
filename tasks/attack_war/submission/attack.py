# Genetic Ver. 2
"""
from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        min_val = 0,
        max_val = 1,
        image_size = [1, 3, 32, 32],
        n_population = 50,
        n_generation = 50,
        mask_rate = 0.2,
        temperature = 0.1,
        use_mask = False,
        step_size = 0.2,
        child_rate = 0.4,
        mutate_rate = 0.8,
        l2_threshold = 12
    ):
        #
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            min_val: minimum value of each element in original image
            max_val: maximum value of each element in original image
                     each element in perturbed image should be in the range of min_val and max_val
            attack_path: Any other sources files that you may want to use like models should be available in ./submissions/ folder and loaded by attack.py.
                         Server doesn't load any external files. Do submit those files along with attack.py
        #
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.min_val = 0
        self.max_val = 1
        self.image_size = image_size
        self.n_population = n_population
        self.n_generation = n_generation
        self.mask_rate = mask_rate
        self.temperature = temperature
        self.use_mask = use_mask
        self.step_size = step_size
        self.child_rate = child_rate
        self.mutate_rate = mutate_rate
        self.l2_threshold = l2_threshold
    def attack(
        self, original_images: torch.tensor, labels: torch.tensor, target_label = None,
    ):
        assert torch.is_tensor(original_images), "original_images should be a torch tensor."
        assert torch.is_tensor(labels), "labels should be a torch tensor."

        original_images = original_images.to(self.device)
        labels = labels.to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        perturbed_image = original_images.detach().cpu().numpy()
        self.mask = np.random.binomial(1, self.mask_rate, size=self.image_size).astype(
            "bool"
        )

        self.original_image = original_images.detach().cpu().numpy()
        #print("o i:", np.shape(self.original_image))
        population = self.init_population(self.original_image)
        examples = [(labels[0], labels[0], np.squeeze(x)) for x in population[:10]]
        success = False
        for g in range(self.n_generation):
            # print("generation: ", g)
            population, output, scores, best_index, detect_outputs = self.eval_population(
                population, target_label
            )
            #if (np.argmax(output[best_index, :]) == target_label) & (detect_outputs[best_index].item() == 0):
            if (np.argmax(output[best_index, :]) == target_label):
                # print(f"Attack Success!")
                perturbed_image = np.array(population[best_index])
                success = True
                break
            # print(type(population))
        perturbed_image = np.array(population[best_index])
        assert type(perturbed_image) == np.ndarray, "perturbed_image should be numpy"
        perturbed_image = torch.FloatTensor(perturbed_image)
        image_tensor = perturbed_image.to(self.device)

        adv_outputs, _ = self.vm.get_batch_output(image_tensor)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        # print("out", g)
        return perturbed_image.detach().cpu().numpy(), correct

    def init_population(self, original_image: np.ndarray):
        #
        Initialize the population to n_population of images. Make sure to perturbe each image.
        args:
            original_image: image to be attacked
        return:
            a list of perturbed images initialized from orignal_image
        #
        return np.array(
            [self.perturb(original_image[0]) for _ in range(self.n_population)]
        )

    def eval_population(self, population, target_label):
        #
        evaluate the population, pick the parents, and then crossover to get the next
        population
        args:
            population: current population, a list of images
            target_label: target label we want the imageto be classiied, int
        return:
            population: population of all the images
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
            best_indx: index of the best image in the population
        #
        output, scores, detect_outputs = self.fitness(population, target_label)
        logits = np.exp(scores / self.temperature)
        select_probs = logits / np.sum(logits)
        score_ranks = np.argsort(scores)[::-1]
        best_index = score_ranks[0]
        # print(detect_outputs.shape, output.shape, best_index)
        #if (np.argmax(output[best_index, :]) == target_label) & (detect_outputs[best_index].item() == 0):
        if (np.argmax(output[best_index, :]) == target_label) & (detect_outputs[best_index] == [0])):
            return population, output, scores, best_index, detect_outputs

        # the elite gene that's defeintely in the next population without perturbation
        elite = [population[best_index]]

        # strong and fit genes passed down to next generation, they have a chance to mutate
        survival_number = int((1 - self.child_rate) * (self.n_population - 1))
        survived = [population[idx] for idx in score_ranks[1 : survival_number + 1]]
        survived = [
            self.perturb(x) if np.random.uniform() < self.mutate_rate else x
            for x in survived
        ]

        # offsprings of strong genes
        child_number = self.n_population - 1 - survival_number
        mom_index = np.random.choice(self.n_population, child_number, p=select_probs)
        dad_index = np.random.choice(self.n_population, child_number, p=select_probs)
        childs = [
            self.crossover(population[mom_index[i]], population[dad_index[i]])
            for i in range(child_number)
        ]
        # childs = [self.perturb(childs[i]) for i in range(len(childs))]
        population = np.array(elite + survived + childs)
        return population, output, scores, best_index, detect_outputs
    def crossover(self, x1, x2):
        #
        crossover two images to get a new one. We use a uniform distribution with p=0.5
        args:
            x1: image #1
            x2: image #2
        return:
            x_new: newly crossovered image
        #
        x_new = x1.copy()
        for i in range(x1.shape[1]):
            for j in range(x1.shape[2]):
                for k in range(x1.shape[3]):
                    if np.random.uniform() < 0.5:
                        x_new[0][i][j][k] = x2[0][i][j][k]
        return x_new

    def fitness(self, image: np.ndarray, target: int):
        #
        evaluate how fit the current image is
        return:
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
        #
        output, detect_outputs = self._get_batch_outputs_numpy(image)
        softmax_output = np.exp(output) / np.expand_dims(
            np.sum(np.exp(output), axis=1), axis=1
        )
        scores = softmax_output[:, target]
        return output, scores, detect_outputs

    def perturb(self, image):
        #
        perturb a single image with some constraints and a mask
        args:
            image: the image to be perturbed
        return:
            perturbed: perturbed image
        #
        if not self.use_mask:
            adv_images = image + np.random.randn(*self.mask.shape) * self.step_size
            # perturbed = np.maximum(np.minimum(adv_images,self.original_image+0.5), self.original_image-0.5)
            delta = np.expand_dims(adv_images - self.original_image, axis=0)
            # Assume x and adv_images are batched tensors where the first dimension is
            # a batch dimension
            eps = self.l2_threshold
            mask = (
                np.linalg.norm(delta.reshape((delta.shape[0], -1)), ord=2, axis=1)
                <= eps
            )
            scaling_factor = np.linalg.norm(
                delta.reshape((delta.shape[0], -1)), ord=2, axis=1
            )
            scaling_factor[mask] = eps
            delta *= eps / scaling_factor.reshape((-1, 1, 1, 1))
            perturbed = self.original_image + delta
            perturbed = np.squeeze(np.clip(perturbed, 0, 1), axis=0)
        else:
            perturbed = np.clip(
                image + self.mask * np.random.randn(*self.mask.shape) * self.step_size,
                0,
                1,
            )
        return perturbed

    def _get_batch_outputs_numpy(self, image: np.ndarray):
        #print("image:", np.shape(image))
        image = np.array([i[0] for i in image])
        image_tensor = torch.FloatTensor(image)
        image_tensor = image_tensor.to(self.device)
        outputs, detect_outputs = self.vm.get_batch_output(image_tensor)
        return outputs.detach().cpu().numpy(), detect_outputs
"""


# Genetic Ver. 1
"""
from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        min_val = 0,
        max_val = 1,
        image_size = [1, 3, 32, 32],
        n_population = 100,
        n_generation = 200,
        mask_rate = 0.2,
        temperature = 0.1,
        use_mask = False,
        step_size = 0.2,
        child_rate = 0.4,
        mutate_rate = 0.8,
        l2_threshold = 12
    ):
        #
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            min_val: minimum value of each element in original image
            max_val: maximum value of each element in original image
                     each element in perturbed image should be in the range of min_val and max_val
            attack_path: Any other sources files that you may want to use like models should be available in ./submissions/ folder and loaded by attack.py.
                         Server doesn't load any external files. Do submit those files along with attack.py
        #
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.min_val = 0
        self.max_val = 1
        self.image_size = image_size
        self.n_population = n_population
        self.n_generation = n_generation
        self.mask_rate = mask_rate
        self.temperature = temperature
        self.use_mask = use_mask
        self.step_size = step_size
        self.child_rate = child_rate
        self.mutate_rate = mutate_rate
        self.l2_threshold = l2_threshold
    def attack(
        self, original_images: torch.tensor, labels: torch.tensor, target_label = None,
    ):
        assert torch.is_tensor(original_images), "original_images should be a torch tensor."
        assert torch.is_tensor(labels), "labels should be a torch tensor."

        original_images = original_images.to(self.device)
        labels = labels.to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        perturbed_image = original_images.detach().cpu().numpy()
        self.mask = np.random.binomial(1, self.mask_rate, size=self.image_size).astype(
            "bool"
        )

        self.original_image = original_images.detach().cpu().numpy()
        #print("o i:", np.shape(self.original_image))
        population = self.init_population(self.original_image)
        examples = [(labels[0], labels[0], np.squeeze(x)) for x in population[:10]]
        success = False
        for g in range(self.n_generation):
            # print("generation: ", g)
            population, output, scores, best_index = self.eval_population(
                population, target_label
            )
            if np.argmax(output[best_index, :]) == target_label:
                # print(f"Attack Success!")
                perturbed_image = np.array(population[best_index])
                success = True
                break
            # print(type(population))
        assert type(perturbed_image) == np.ndarray, "perturbed_image should be numpy"
        perturbed_image = torch.FloatTensor(perturbed_image)
        image_tensor = perturbed_image.to(self.device)

        adv_outputs, _ = self.vm.get_batch_output(image_tensor)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        return perturbed_image.detach().cpu().numpy(), correct

    def init_population(self, original_image: np.ndarray):
        #
        Initialize the population to n_population of images. Make sure to perturbe each image.
        args:
            original_image: image to be attacked
        return:
            a list of perturbed images initialized from orignal_image
        #
        return np.array(
            [self.perturb(original_image[0]) for _ in range(self.n_population)]
        )

    def eval_population(self, population, target_label):
        output, scores = self.fitness(population, target_label)
        score_ranks = np.sort(scores, axis=None)[::-1]
        best_index = np.where(scores == score_ranks[0])[0][0]
        logits = np.exp(scores / self.temperature)
        select_probs = logits / np.sum(logits)


        if np.argmax(output[best_index, :]) == target_label:
            return population, output, scores, best_index

        elite = [population[best_index]]

        offspring = int((self.n_population - 1) * self.child_rate)

        survive = self.n_population - 1 - offspring

        survived = []

        for i in range(1, survive + 1):
            sind = np.where(scores == score_ranks[i])[0][0]
            survived.append(population[sind])

        mutate_num = int(survive * self.mutate_rate)

        randomchoice = np.random.choice(survive, mutate_num, replace=False)


        for i in range(0, mutate_num - 1):
            survived[randomchoice[i]] = self.perturb(survived[randomchoice[i]])

        ptemp = select_probs[0:40]
        pselect = ptemp / np.sum(ptemp)
        children = []
        for i in range(0, 100 - 1 - survive):
            select = np.random.choice(40, 2, replace=False, p=pselect)
            sind1 = np.where(scores == score_ranks[select[0]])[0][0]
            sind2 = np.where(scores == score_ranks[select[1]])[0][0]
            pimage = self.crossover(population[sind1], population[sind2])
            children.append(pimage)

        population = np.array(elite + survived + children)
        return population, output, scores, best_index

    def crossover(self, x1, x2):
        #
        crossover two images to get a new one. We use a uniform distribution with p=0.5
        args:
            x1: image #1
            x2: image #2
        return:
            x_new: newly crossovered image
        #
        x_new = x1.copy()
        for i in range(x1.shape[1]):
            for j in range(x1.shape[2]):
                for k in range(x1.shape[3]):
                    if np.random.uniform() < 0.5:
                        x_new[0][i][j][k] = x2[0][i][j][k]
        return x_new

    def fitness(self, image: np.ndarray, target: int):
        #
        evaluate how fit the current image is
        return:
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
        #
        output = self._get_batch_outputs_numpy(image)
        softmax_output = np.exp(output) / np.expand_dims(
            np.sum(np.exp(output), axis=1), axis=1
        )
        scores = softmax_output[:, target]
        return output, scores


    def perturb(self, image):
        #
        perturb a single image with some constraints and a mask
        args:
            image: the image to be perturbed
        return:
            perturbed: perturbed image
        #
        if not self.use_mask:
            adv_images = image + np.random.randn(*self.mask.shape) * self.step_size
            # perturbed = np.maximum(np.minimum(adv_images,self.original_image+0.5), self.original_image-0.5)
            delta = np.expand_dims(adv_images - self.original_image, axis=0)
            # Assume x and adv_images are batched tensors where the first dimension is
            # a batch dimension
            eps = self.l2_threshold
            mask = (
                np.linalg.norm(delta.reshape((delta.shape[0], -1)), ord=2, axis=1)
                <= eps
            )
            scaling_factor = np.linalg.norm(
                delta.reshape((delta.shape[0], -1)), ord=2, axis=1
            )
            scaling_factor[mask] = eps
            delta *= eps / scaling_factor.reshape((-1, 1, 1, 1))
            perturbed = self.original_image + delta
            perturbed = np.squeeze(np.clip(perturbed, 0, 1), axis=0)
        else:
            perturbed = np.clip(
                image + self.mask * np.random.randn(*self.mask.shape) * self.step_size,
                0,
                1,
            )
        return perturbed

    def _get_batch_outputs_numpy(self, image: np.ndarray):
        #print("image:", np.shape(image))
        image = np.array([i[0] for i in image])
        image_tensor = torch.FloatTensor(image)
        image_tensor = image_tensor.to(self.device)

        outputs, _ = self.vm.get_batch_output(image_tensor)

        return outputs.detach().cpu().numpy()
"""


# PGD 
"""
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Attack:
    # PGD Attack
    def __init__(self, vm, device, attack_path, epsilons=[1,2,4,8,16], alpha=0.1, min_val=0, max_val=1, max_iters=50,  _type='l2'):
        # self.model = model._to(device)
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.epsilons = epsilons
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha
        self.max_iters = max_iters
        self._type = _type

    def project(self, x, original_x, epsilon, _type='l2'):
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

    def attack(self, original_images, labels, target_label = None, reduction4loss='mean', random_start=True):
        self.vm.defender.model.eval()
        assert len(labels) == 1, "image and label size should be 1. Otherwise, consider attack_batch function."
        assert torch.is_tensor(original_images), "original_images should be a torch tensor."
        assert torch.is_tensor(labels), "labels should be a torch tensor."
        assert target_label != None, "target label should not be None"
        for epsilon in self.epsilons:
            self.epsilon = epsilon
            # print(self.epsilon)
            original_images = original_images.to(self.device)
            labels = labels.to(self.device)
            target_labels = target_label * torch.ones_like(labels).to(self.device)

            if random_start:
                rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                    -self.epsilon, self.epsilon)
                rand_perturb = rand_perturb.to(self.device)
                x = original_images + rand_perturb
                x.clamp_(self.min_val, self.max_val)
            else:
                x = original_images.clone()

            with torch.enable_grad():
                for _iter in range(self.max_iters):
                    grads = self.vm.get_batch_input_gradient(x.data, target_labels)
                    x.data -= self.alpha * torch.sign(grads.data)
                    x = self.project(x, original_images, self.epsilon, self._type)
                    x.clamp_(self.min_val, self.max_val)
                    outputs, detect_outputs = self.vm.get_batch_output(x)
                    # print(detect_outputs.item())
                    if detect_outputs == [1]:
                        # print("detect_outputs")
                        continue
                    elif detect_outputs == [0]:
                        final_pred = outputs.max(1, keepdim=True)[1]
                        # print(final_pred, target_labels )
                        correct = (final_pred == target_labels).sum().item()
                        if correct == 1:
                            # print("boom 0")
                            return x.cpu().detach().numpy(), correct
                    
        return x.cpu().detach().numpy(), 0

    def attack_batch(self, original_images, labels, target_label = None, reduction4loss='mean', random_start=True):
        assert torch.is_tensor(labels), "labels should be a torch tensor."
        for epsilon in self.epsilons:
            self.epsilon = epsilon
            original_images = original_images.to(self.device)
            labels = labels.to(self.device)
            target_labels = target_label * torch.ones_like(labels).to(self.device)

            if random_start:
                rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                    -self.epsilon, self.epsilon)
                rand_perturb = rand_perturb.to(self.device)
                x = original_images + rand_perturb
                x.clamp_(self.min_val, self.max_val)
            else:
                x = original_images.clone()

            with torch.enable_grad():
                for _iter in range(self.max_iters):
                    grads = self.vm.get_batch_input_gradient(x.data, target_labels)
                    x.data -= self.alpha * torch.sign(grads.data)
                    x = self.project(x, original_images, self.epsilon, self._type)
                    x.clamp_(self.min_val, self.max_val)
                    outputs, detect_outputs = self.vm.get_batch_output(x)
        return x.cpu().detach().numpy(), 0
"""


# Original Attack
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
        #
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            epsilon: magnitude of perturbation that is added
        #
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




# Genetic Ver. 3

from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        min_val = 0,
        max_val = 1,
        image_size = [1, 3, 32, 32],
        n_population = 50,  #100
        n_generation = 50,  #200
        mask_rate = 0.2,
        temperature = 0.1,
        use_mask = False,
        step_size = 0.2,
        child_rate = 0.4,
        mutate_rate = 0.8,
        l2_threshold = 12
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
        self.min_val = 0
        self.max_val = 1
        self.image_size = image_size
        self.n_population = n_population
        self.n_generation = n_generation
        self.mask_rate = mask_rate
        self.temperature = temperature
        self.use_mask = use_mask
        self.step_size = step_size
        self.child_rate = child_rate
        self.mutate_rate = mutate_rate
        self.l2_threshold = l2_threshold
    def attack(
        self, original_images: torch.tensor, labels: torch.tensor, target_label = None,
    ):
        assert torch.is_tensor(original_images), "original_images should be a torch tensor."
        assert torch.is_tensor(labels), "labels should be a torch tensor."

        original_images = original_images.to(self.device)
        labels = labels.to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        perturbed_image = original_images.detach().cpu().numpy()
        self.mask = np.random.binomial(1, self.mask_rate, size=self.image_size).astype(
            "bool"
        )

        self.original_image = original_images.detach().cpu().numpy()
        #print("o i:", np.shape(self.original_image))
        population = self.init_population(self.original_image)
        examples = [(labels[0], labels[0], np.squeeze(x)) for x in population[:10]]
        success = False
        for g in range(self.n_generation):
            # print("generation: ", g)
            population, output, scores, best_index, detect_outputs = self.eval_population(
                population, target_label
            )
            #if (np.argmax(output[best_index, :]) == target_label) & (detect_outputs[best_index].item() == 0):
            #if (np.argmax(output[best_index, :]) == target_label) & (detect_outputs[best_index] == 0):
            if (np.argmax(output[best_index, :]) == target_label):
                # print(f"Attack Success!")
                perturbed_image = np.array(population[best_index])
                success = True
                break
            # print(type(population))
        perturbed_image = np.array(population[best_index])
        assert type(perturbed_image) == np.ndarray, "perturbed_image should be numpy"
        perturbed_image = torch.FloatTensor(perturbed_image)
        image_tensor = perturbed_image.to(self.device)

        adv_outputs, _ = self.vm.get_batch_output(image_tensor)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        # print("out", g)
        return perturbed_image.detach().cpu().numpy(), correct

    def init_population(self, original_image: np.ndarray):
        """
        Initialize the population to n_population of images. Make sure to perturbe each image.
        args:
            original_image: image to be attacked
        return:
            a list of perturbed images initialized from orignal_image
        """
        return np.array(
            [self.perturb(original_image[0]) for _ in range(self.n_population)]
        )

    def eval_population(self, population, target_label):
        """
        evaluate the population, pick the parents, and then crossover to get the next
        population
        args:
            population: current population, a list of images
            target_label: target label we want the imageto be classiied, int
        return:
            population: population of all the images
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
            best_indx: index of the best image in the population
        """
        output, scores, detect_outputs = self.fitness(population, target_label)
        logits = np.exp(scores / self.temperature)
        select_probs = logits / np.sum(logits)
        score_ranks = np.argsort(scores)[::-1]
        best_index = score_ranks[0]
        #print(detect_outputs.shape, output.shape, best_index)
        #if (np.argmax(output[best_index, :]) == target_label) & (detect_outputs[best_index].item() == 0):
        #if (np.argmax(output[best_index, :]) == target_label) & (detect_outputs[best_index] == 0):
        if (np.argmax(output[best_index, :]) == target_label):
                        return population, output, scores, best_index, detect_outputs

        # the elite gene that's defeintely in the next population without perturbation
        elite = [population[best_index]]

        # strong and fit genes passed down to next generation, they have a chance to mutate
        survival_number = int((1 - self.child_rate) * (self.n_population - 1))
        survived = [population[idx] for idx in score_ranks[1 : survival_number + 1]]
        survived = [
            self.perturb(x) if np.random.uniform() < self.mutate_rate else x
            for x in survived
        ]

        # offsprings of strong genes
        child_number = self.n_population - 1 - survival_number
        mom_index = np.random.choice(self.n_population, child_number, p=select_probs)
        dad_index = np.random.choice(self.n_population, child_number, p=select_probs)
        childs = [
            self.crossover(population[mom_index[i]], population[dad_index[i]])
            for i in range(child_number)
        ]
        # childs = [self.perturb(childs[i]) for i in range(len(childs))]
        population = np.array(elite + survived + childs)
        return population, output, scores, best_index, detect_outputs
    def crossover(self, x1, x2):
        """
        crossover two images to get a new one. We use a uniform distribution with p=0.5
        args:
            x1: image #1
            x2: image #2
        return:
            x_new: newly crossovered image
        """
        x_new = x1.copy()
        for i in range(x1.shape[1]):
            for j in range(x1.shape[2]):
                for k in range(x1.shape[3]):
                    if np.random.uniform() < 0.5:
                        x_new[0][i][j][k] = x2[0][i][j][k]
        return x_new


    def softmax(self, x):
        r=np.exp(x - np.max(x))
        return r/r.sum(axis=1).reshape(x.shape[0], 1)

    def fitness(self, image: np.ndarray, target: int):
        """
        evaluate how fit the current image is
        return:
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
        """
        output, detect_outputs = self._get_batch_outputs_numpy(image)
        softmax_output = self.softmax(output)
        scores = softmax_output[:, target]
        return output, scores, detect_outputs

    def perturb(self, image):
        """
        perturb a single image with some constraints and a mask
        args:
            image: the image to be perturbed
        return:
            perturbed: perturbed image
        """
        if not self.use_mask:
            adv_images = image + np.random.randn(*self.mask.shape) * self.step_size
            # perturbed = np.maximum(np.minimum(adv_images,self.original_image+0.5), self.original_image-0.5)
            delta = np.expand_dims(adv_images - self.original_image, axis=0)
            # Assume x and adv_images are batched tensors where the first dimension is
            # a batch dimension
            eps = self.l2_threshold
            mask = (
                np.linalg.norm(delta.reshape((delta.shape[0], -1)), ord=2, axis=1)
                <= eps
            )
            scaling_factor = np.linalg.norm(
                delta.reshape((delta.shape[0], -1)), ord=2, axis=1
            )
            scaling_factor[mask] = eps
            delta *= eps / scaling_factor.reshape((-1, 1, 1, 1))
            perturbed = self.original_image + delta
            perturbed = np.squeeze(np.clip(perturbed, 0, 1), axis=0)
        else:
            perturbed = np.clip(
                image + self.mask * np.random.randn(*self.mask.shape) * self.step_size,
                0,
                1,
            )
        return perturbed

    def _get_batch_outputs_numpy(self, image: np.ndarray):
        #print("image:", np.shape(image))
        image = np.array([i[0] for i in image])
        image_tensor = torch.FloatTensor(image)
        image_tensor = image_tensor.to(self.device)
        outputs, detect_outputs = self.vm.get_batch_output(image_tensor)
        return outputs.detach().cpu().numpy(), detect_outputs

