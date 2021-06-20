import os
import time
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models.vgg_for_feature_extraction import vgg
from models.resnet_for_feature_extraction import resnet
from utils.dataset import get_CelebA_dataset
from tqdm import tqdm

'''
Steps to visualize the features of intermediate layers
1. get the feature map of the given intermediate layer
2. reduce the dimension of the feature to 2 by PCA or t-SNE
3. visualize the results
'''

def reduce_dimension_by_pca(features:np.ndarray, n_components=2):
    features = features.reshape(features.shape[0], -1)
    return PCA(n_components=n_components).fit_transform(features)

def reduce_dimension_by_tsne(features:np.ndarray, n_components=2):
    features = features.reshape(features.shape[0], -1)
    return TSNE(n_components=n_components).fit_transform(features)


def visualize_2D(data, labels, name=''):
    assert data.ndim == 2 and labels.ndim == 1
    assert data.shape[0] == len(labels) and data.shape[1] == 2
    plt.cla()
    plt.scatter(data[:, 0], data[:, 1], c=labels, marker='o', s=25, cmap=plt.cm.Set1)
    plt.savefig(f'results/feature_analysis/{name}_{time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())}.png')

def load_three_linear_vgg(model, state_dict):
    mapping = {
        'classifier.1.weight': 'linear1.1.weight',
        'classifier.1.bias': 'linear1.1.bias',
        'classifier.4.weight': 'linear2.0.weight',
        'classifier.4.bias': 'linear2.0.bias',
        'classifier.7.weight': 'linear3.weight',
        'classifier.7.bias': 'linear3.bias',
    }

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'classifier' not in k:
            new_state_dict[k] = v
        else:
            new_state_dict[mapping[k]] = v
    # load params
    model.load_state_dict(new_state_dict)

    return model

if __name__ == '__main__':
    # Load datasets
    batch_size = 64
    n_batches = 10
    target_category = 20
    _, val_dataset, _ = get_CelebA_dataset(path='/state/partition/zdzhu/')
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=8, pin_memory=True)

    # Load vgg model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'resnet'
    if model_name == 'vgg':
        model = vgg(layers=16, num_classes=40).to(device)
        model.eval()
        checkpoing_root = 'states/VGG16_0.0001_19_12_47_16'
        checkpoint_lists = [
            # os.path.join(checkpoing_root, 'checkpoint_VGG_0.tar'),
            # os.path.join(checkpoing_root, 'checkpoint_VGG_5.tar'),
            os.path.join(checkpoing_root, 'checkpoint_VGG_13.tar')
        ]
        # name_list = [0, 5, 13]
        name_list = [13]
    elif model_name == 'resnet':
        model = resnet(layers=34, num_classes=40).to(device)
        model.eval()
        checkpoing_root = 'states/resnet34_2021_06_09_09_07_41'
        checkpoint_lists = [
            os.path.join(checkpoing_root, 'checkpoint_resnet34_0.tar'),
            os.path.join(checkpoing_root, 'checkpoint_resnet34_5.tar'),
            os.path.join(checkpoing_root, 'checkpoint_resnet34_15.tar')
        ]
        name_list = [0, 5, 15]
    else:
        raise NotImplementedError


    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharex=True, sharey=True)
    for i, checkpoint_path in enumerate(checkpoint_lists):
        checkpoint = torch.load(checkpoint_path)

        # resnet
        model.load_state_dict(checkpoint['state_dict'])
        features = [[] for _ in range(4)]
        labels = []

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(val_dataloader):
                if batch_idx == n_batches:
                    break
                input = input.to(device)
                _, f1, f2, f3, f4 = model(input)
                features[0].append(f1.detach().cpu().numpy())
                features[1].append(f2.detach().cpu().numpy())
                features[2].append(f3.detach().cpu().numpy())
                features[3].append(f4.detach().cpu().numpy())
                labels.append(target[:, target_category])

        labels = np.vstack(labels)
        l0 = np.vstack(features[0])
        l1 = np.vstack(features[1])
        l2 = np.vstack(features[2])
        l3 = np.vstack(features[3])

        f = [l0, l1, l2, l3]

        # vgg linear
        # model = load_three_linear_vgg(model, checkpoint['state_dict'])
        # features = [[] for _ in range(4)]
        # labels = []
        #
        # with torch.no_grad():
        #     for batch_idx, (input, target) in enumerate(val_dataloader):
        #         if batch_idx == n_batches:
        #             break
        #         input = input.to(device)
        #         out_linear3, out_conv, out_linear1, out_linear2 = model(input)
        #         features[0].append(out_conv.detach().cpu().numpy())
        #         features[1].append(out_linear1.detach().cpu().numpy())
        #         features[2].append(out_linear2.detach().cpu().numpy())
        #         features[3].append(out_linear3.detach().cpu().numpy())
        #         labels.append(target[:, target_category])
        #
        # labels = np.vstack(labels)
        # l0 = np.vstack(features[0])
        # l1 = np.vstack(features[1])
        # l2 = np.vstack(features[2])
        # l3 = np.vstack(features[3])
        #
        # f = [l0, l1, l2]

        for j, feature in tqdm(enumerate(f)):
            print(feature.shape)
            # feature = reduce_dimension_by_pca(feature, n_components=128)
            reduced_features = reduce_dimension_by_tsne(feature, n_components=2)
            axes[i, j].scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, marker='o', s=25, cmap=plt.cm.Set1)
            # axes[i, j].axis('off')
            axes[i, j].xaxis.set_visible(False)
            axes[i, j].yaxis.set_visible(False)
            axes[i, j].set_title(f"Layer {j}, epoch {name_list[i]}")
            # axes[i, j].set_title(f"Layer {j}")

    plt.grid(True)
    plt.savefig('docs/figs/tsne.png')