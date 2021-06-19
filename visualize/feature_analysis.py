import time
import torch
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models.vgg import vgg16_bn
from models.resnet_for_feature_analysis import resnet50
from utils.dataset import get_CelebA_dataset

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


if __name__ == '__main__':
    # parameter setting
    n_batches = 10
    dim_to_vis = 0

    # Load vgg model
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    model = vgg16_bn(num_classes=40).to(device)
    model.eval()
    checkpoint = torch.load('../states/vgg_2021_06_10_15_44_29/checkpoint_vgg_best.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # Load model and checkpoints
    # device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    # model = resnet50(num_classes=40).to(device)
    # checkpoint = torch.load('../states/resnet_2021_06_09_00_57_56/checkpoint_resnet_2.tar')
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()

    # load dataset
    _, val_dataset, _ = get_CelebA_dataset('/state/partition/zdzhu')
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False,num_workers=8)
    labels = []
    output_list = []
    intermediate_feature_1_list, intermediate_feature_2_list, intermediate_feature_3_list, intermediate_feature_4_list = [], [], [], []

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_dataloader):
            if batch_idx == n_batches:
                break
            input = input.to(device)
            output, intermediate_feature_1, intermediate_feature_2, intermediate_feature_3, intermediate_feature_4 = model(input)
            output_list.append(output.detach().cpu().numpy())
            intermediate_feature_1_list.append(intermediate_feature_1.detach().cpu().numpy())
            intermediate_feature_2_list.append(intermediate_feature_2.detach().cpu().numpy())
            intermediate_feature_3_list.append(intermediate_feature_3.detach().cpu().numpy())
            intermediate_feature_4_list.append(intermediate_feature_4.detach().cpu().numpy())
            labels.append(target.numpy())

    labels = np.vstack(labels)
    output_list = np.vstack(output_list)
    intermediate_feature_1_list = np.vstack(intermediate_feature_1_list)
    intermediate_feature_2_list = np.vstack(intermediate_feature_2_list)
    intermediate_feature_3_list = np.vstack(intermediate_feature_3_list)
    intermediate_feature_4_list = np.vstack(intermediate_feature_4_list)

    # features = reduce_dimension_by_pca(intermediate_feature_1_list, n_components=2)
    # visualize_2D(features, labels[:, dim_to_vis], name='pca_1')
    #
    # features = reduce_dimension_by_pca(intermediate_feature_2_list, n_components=2)
    # visualize_2D(features, labels[:, dim_to_vis], name='pca_2')
    #
    # features = reduce_dimension_by_pca(intermediate_feature_3_list, n_components=2)
    # visualize_2D(features, labels[:, dim_to_vis], name='pca_3')
    #
    # features = reduce_dimension_by_pca(intermediate_feature_4_list, n_components=2)
    # visualize_2D(features, labels[:, dim_to_vis], name='pca_4')


    def tsne():
        features = reduce_dimension_by_tsne(intermediate_feature_1_list, n_components=2)
        visualize_2D(features, labels[:, dim_to_vis], name='tsne_1')

        features = reduce_dimension_by_tsne(intermediate_feature_2_list, n_components=2)
        visualize_2D(features, labels[:, dim_to_vis], name='tsne_2')

        features = reduce_dimension_by_tsne(intermediate_feature_3_list, n_components=2)
        visualize_2D(features, labels[:, dim_to_vis], name='tsne_3')

        features = reduce_dimension_by_tsne(intermediate_feature_4_list, n_components=2)
        visualize_2D(features, labels[:, dim_to_vis], name='tsne_4')

    tsne()