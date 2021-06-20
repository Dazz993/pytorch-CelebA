import os
import cv2
import time
import yaml
import time
import argparse
import torch
import torch.utils.data
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from tqdm import tqdm
from time import perf_counter as t
from models import alexnet, vgg, resnet
from utils.dataset import get_CelebA_dataset
import torchvision.utils as tuitls

def cam(model, input_tensor, ori_image, target_layer, target_category, write=True):
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(ori_image, grayscale_cam, use_rgb=True)

    if write:
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        ori_image = cv2.cvtColor(np.uint8(ori_image * 255), cv2.COLOR_RGB2BGR)
        ori_image = cv2.resize(ori_image, (178, 218))
        cam_image = cv2.resize(cam_image, (178, 218))
        cv2.imwrite(f'docs/figs/ori.png', ori_image)
        cv2.imwrite(f'docs/figs/cam.png', cam_image)

    return cam_image

def cam_batch(model, input_tensor, imgs, target_layer, target_category, n_row):
    cam = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    images = []
    for i, img in tqdm(enumerate(imgs)):
        cam_image = show_cam_on_image(img, grayscale_cam[i], use_rgb=True)
        cam_image = cam_image.transpose(2, 0, 1)
        images.append(cam_image)

    images = torch.FloatTensor(np.array(images))

    print(images.max())

    tuitls.save_image(images.data, f'docs/figs/cam_batch.png', normalize=True, nrow=n_row)



def guided_propagation(model, input_tensor, target_category, write=True):
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=False)
    gb = gb_model(input_tensor, target_category=target_category)
    gb = _deprocess_image(gb)

    if write:
        gb = cv2.resize(gb, (178, 218))
        cv2.imwrite(f'docs/figs/gb.png', gb)

    return gb

def gb_batch(model, input_tensor, target_layer, target_category, n_row):
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=False)
    images = []
    for _input_tensor in tqdm(input_tensor):
        _input_tensor = _input_tensor.unsqueeze(0)
        gb = gb_model(_input_tensor, target_category=target_category)
        gb = _deprocess_image(gb.transpose(2, 0, 1))
        images.append(gb)

    images = torch.FloatTensor(np.array(images))

    print(images.max())
    tuitls.save_image(images.data, f'docs/figs/gb_batch.png', normalize=True, nrow=n_row)

def _deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

if __name__ == '__main__':
    # define some parameters
    device = 'cpu'
    IMG_BEGIN = 162771
    ouput_width = 6
    img_index = 162771
    tensor_index = img_index - IMG_BEGIN
    target_category = 5

    # load dataset
    _, val_dataset, _ = get_CelebA_dataset('/state/partition/zdzhu')
    image_folder = os.path.join(val_dataset.root, val_dataset.base_folder, "img_align_celeba")
    filename = val_dataset.filename

    if target_category is not None:
        print(val_dataset.attr_names[target_category])

    input_tensor_list = []
    imgs = []

    for i in range(ouput_width ** 2):
        input_tensor = val_dataset[tensor_index + i][0]

        # load image
        img = cv2.imread(os.path.join(image_folder, filename[tensor_index + i]), 1)[:, :, ::-1]
        img = np.float32(img) / 255
        img = cv2.resize(img, (224, 224))

        input_tensor_list.append(input_tensor)
        imgs.append(img)

    input_tensor = torch.stack(input_tensor_list)
    imgs = np.stack(imgs)

    # Simple Conv
    # model = SimpleConv(num_classes=10).to(device)
    # state_dicts = torch.load('states/SimpleConv_0.001__2021_06_17_22_42_13/checkpoint_SimpleConv_best.tar')
    # model.load_state_dict(state_dicts['state_dict'])
    # target_layer = model.features[-1]

    # Alexnet
    # model = alexnet(num_classes=40).to(device)
    # state_dicts = torch.load('states/Alexnet_0.0001_19_12_05_07/checkpoint_Alexnet_best.tar', map_location={'cuda:0': 'cuda:1'})
    # model.load_state_dict(state_dicts['state_dict'])
    # target_layer = model.features[-1]

    # VGG 16
    # model = vgg(layers=16, num_classes=40).to(device)
    # state_dicts = torch.load('states/VGG16_0.0001_19_12_47_16/checkpoint_VGG_best.tar', map_location={'cuda:0': 'cuda:1'})
    # model.load_state_dict(state_dicts['state_dict'])
    # target_layer = model.features[-1]

    # VGG 19
    # model = vgg(layers=19, num_classes=40).to(device)
    # state_dicts = torch.load('states/VGG19_0.0001_20_01_12_28/checkpoint_VGG_17.tar', map_location={'cuda:0': 'cuda:1'})
    # model.load_state_dict(state_dicts['state_dict'])
    # target_layer = model.features[-1]

    # ResNet 34
    # model = resnet(layers=34, num_classes=40).to(device)
    # state_dicts = torch.load('states/resnet34_2021_06_09_09_07_41/checkpoint_resnet34_best.tar')
    # model.load_state_dict(state_dicts['state_dict'])
    # target_layer = model.layer4[-1]

    # ResNet 101
    # model = resnet(layers=101, num_classes=40).to(device)
    # state_dicts = torch.load('states/resnet101_2021_06_09_11_23_30/checkpoint_resnet101_best.tar', map_location={'cuda:0': 'cuda:1'})
    # model.load_state_dict(state_dicts['state_dict'])
    # target_layer = model.layer4[-1]

    # ResNet 152
    model = resnet(layers=152, num_classes=40)
    state_dicts = torch.load('states/resnet152_2021_06_09_11_33_18/checkpoint_resnet152_best.tar', map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(state_dicts['state_dict'])
    target_layer = model.layer4[-1]


    model.eval()

    # cam(model, input_tensor=input_tensor[0].unsqueeze(0), ori_image=imgs[0], target_layer=target_layer, target_category=target_category)

    # guided_propagation(model=model, input_tensor=input_tensor, target_category=None)

    cam_batch(model=model, input_tensor=input_tensor, imgs=imgs, target_layer=target_layer, target_category=target_category, n_row=ouput_width)

    gb_batch(model=model, input_tensor=input_tensor, target_layer=target_layer,
              target_category=target_category, n_row=ouput_width)