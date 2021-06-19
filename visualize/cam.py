import os
import cv2
import time
import torch
import torch.utils.data
import numpy as np
from models.vgg import vgg16_bn
from models.resnet import resnet50
from utils.dataset import get_CelebA_dataset
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


if __name__ == '__main__':
    # parameter setting
    IMG_BEGIN = 162771
    img_index = 163220
    tensor_index = img_index - IMG_BEGIN

    # Load vgg model
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    model = vgg16_bn(pretrained=False, num_classes=40).to(device)
    model.eval()
    checkpoint = torch.load('../states/vgg_2021_06_10_15_44_29/checkpoint_vgg_best.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # Load model and checkpoints
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # model = resnet50(num_classes=40).to(device)
    # checkpoint = torch.load('../states/resnet_2021_06_09_00_57_56/checkpoint_resnet_best.tar')
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()

    # define target layer and target category
    # target_layer = model.layer4[-2]
    target_layer = model.features[-1]
    target_category = 15

    print(f"Tensor index: {tensor_index}, target_category: {target_category}")

    # load dataset
    _, val_dataset, _ = get_CelebA_dataset('/state/partition/zdzhu')
    image_folder = os.path.join(val_dataset.root, val_dataset.base_folder, "img_align_celeba")
    filename = val_dataset.filename
    input_tensor = val_dataset[tensor_index][0].unsqueeze(0)

    print(input_tensor.shape)

    print(f"Ground truth: {val_dataset[tensor_index][1][target_category]}")
    print(f"Model result: {model(input_tensor.to(device))[0][target_category].item()}")

    # load image
    img = cv2.imread(os.path.join(image_folder, filename[tensor_index]), 1)[:, :, ::-1]
    img = np.float32(img) / 255
    img = cv2.resize(img, (224, 224))

    cam = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]

    print(grayscale_cam.shape)

    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cam_image = cv2.resize(cam_image, (178, 218))
    cv2.imwrite(f'results/cam/{time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())}.png', cam_image)