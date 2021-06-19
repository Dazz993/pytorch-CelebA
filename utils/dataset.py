import os
import numpy as np
import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

path = '../data/'

def get_dataset(name, path):
    if name == 'CelebA':
        return get_CelebA_dataset(path)
    else:
        raise NotImplementedError(f'Getting {name} dataset is not implemented')

def get_CelebA_dataset(path):
    normalize = transforms.Normalize(mean=[0.506, 0.425, 0.383],
                                     std=[0.309, 0.289, 0.288])

    train_dataset = datasets.CelebA(root=path, split='train', target_type='attr', transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]), download=False)

    val_dataset = datasets.CelebA(root=path, split='valid', target_type='attr', transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]), download=False)

    test_dataset = datasets.CelebA(root=path, split='test', target_type='attr', transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]), download=False)

    return train_dataset, val_dataset, test_dataset