import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Data Preparation')
parser.add_argument('--data-dir', default='', type=str, metavar='PATH',
                    help='path to the image folders')

def main():
    args = parser.parse_args()
    means, stds = compute_image_parameters(args.data_dir)

    print(means, stds)


def compute_image_parameters(path, resize_to=(224, 224) ,sample_rate=0.1):
    '''
    Compute means and standard deviations of images

    Args:
        path (str): path to the image folder
        resize_to (tuple, optional): the size we want to resize to. Defaults to (224, 224).
        sample_rate (float, optional): sample rate to compute. Defaults to 0.1.

    Returns:
        means (list of floats): means value in (0, 1)
        stds (list of floats): stds value in (0, 1)
    '''
    files = os.listdir(path)
    img_files = [file for file in files if len(file) > 4 and file[-4:] == '.jpg']
    print(f"==> Loading images. file path: {path}, num of images: {len(img_files)}")

    imgs, means, stds = [], [], []

    # read imgs
    for img_name in tqdm(random.sample(img_files, int(len(img_files) * sample_rate))):
        img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, resize_to)
        img = img[:, :, :, np.newaxis]
        imgs.append(img)

    # calculate means and stdevs
    imgs = np.array(imgs).astype(np.float32) / 255
    print(f"==> Calculating means and standard deviations")
    for i in tqdm(range(3)):
        means.append(np.mean(imgs[:, :, :, i]))
        stds.append(np.std(imgs[:, :, :, i]))

    # reverse means and stdevs (BGR -> RGB)
    means.reverse()
    stds.reverse()

    return means, stds

if __name__ == '__main__':
    main()