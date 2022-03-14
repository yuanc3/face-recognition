#coding:utf-8
# 路径置顶
import sys
sys.path.append("/home/aistudio/external-libraries/")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.getcwd())
from torch.nn.modules.distance import PairwiseDistance
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import time
# 导入文件
from train_dataset import TrainDataset
from dataset_lfw import TestDataset
from triplet_loss import TripletLoss
import torchvision.models as models
import torchvision.transforms as transforms
# from models import Resnet18Triplet
import torch
from eval_lfw_tool import *

config = {'name':'config'}
config['test_pairs_paths'] = 'Datasets/test_pairs.npy'
config['LFW_data_path'] = 'Datasets/lfw-deepfunneled'
config['LFW_pairs'] = 'Datasets/lfw_pairs.txt'
config['predicter_path'] = 'shape_predictor_68_face_landmarks.dat'
config['image_size']=256
config['test_batch_size'] = 30
config['num_workers']=0


class Resnet18Triplet(nn.Module):
    """Constructs a ResNet-18 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 128.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=128, pretrained=False):
        super(Resnet18Triplet, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        input_features_fc_layer = self.model.fc.in_features
        # Output embedding
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)

    def l2_norm(self, input):
        """Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
        """
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization and multiplication
        by scalar (alpha)."""
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        #   Equation 9: number of classes in VGGFace2 dataset = 9131
        #   lower bound on alpha = 5, multiply alpha by 2; alpha = 10
        alpha = 10
        embedding = embedding * alpha

        return embedding
# 测试数据的变换
test_data_transforms = transforms.Compose([
    # transforms.Resize([config['image_size'], config['image_size']]), # resize
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
        )
    ])
# 测试数据生成器
dataset=TestDataset(
            dir=config['LFW_data_path'],
            pairs_path=config['LFW_pairs'],
            predicter_path=config['predicter_path'],
            img_size=config['image_size'],
            transform=test_data_transforms,
            test_pairs_paths=config['test_pairs_paths']
        )
dataset_1 = dataset
test_dataloader = torch.utils.data.DataLoader(
        dataset_1,
        batch_size=config['test_batch_size'],
        num_workers=config['num_workers'],
        shuffle=False)
#for index,(img1,img2,issame) in enumerate(test_dataloader):
#    print(img1.shape)

# 模型加载
model = Resnet18Triplet(pretrained=False,embedding_dimension = 128)
if torch.cuda.is_available():
    model.cuda()
    print('Using single-gpu testing.')

model_pathi="../famous-enterprises-fr/week5/Model_training_checkpoints/model_resnet18_triplet_epoch_603.pt"
if os.path.exists(model_pathi):
    model_state = torch.load(model_pathi)
    model.load_state_dict(model_state['model_state_dict'])
    start_epoch = model_state['epoch']
    print('loaded %s' % model_pathi)
else:
    print('不存在预训练模型！')


l2_distance = PairwiseDistance(2)
with torch.no_grad():  # 不传梯度了
    distances, labels = [], []
    progress_bar = enumerate(tqdm(test_dataloader))
    for batch_index, (data_a, data_b, label) in progress_bar:
    #for batch_index, (data_a, data_b, label) in enumerate(test_dataloader):
        # data_a, data_b, label这仨是一批的矩阵
        data_a = data_a.cuda()
        data_b = data_b.cuda()
        label = label.cuda()
        output_a, output_b = model(data_a), model(data_b)
        output_a = torch.div(output_a, torch.norm(output_a))
        output_b = torch.div(output_b, torch.norm(output_b))
        distance = l2_distance.forward(output_a, output_b)
        # 列表里套矩阵
        labels.append(label.cpu().detach().numpy())
        distances.append(distance.cpu().detach().numpy())
        #if batch_index >=3:
        #    break
    print("get all image's distance done")
    
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for distance in distances for subdist in distance])
    true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
    tar, far = evaluate_lfw(
        distances=distances,
        labels=labels,
        epoch='',
        tag='NOTMaskedLFW_aucnotmask_valid',
        version="20201102",
        pltshow=False
    )

# 打印日志内容
print('LFW_test_log:\tAUC: {:.3f}\tACC: {:.3f}+-{:.3f}\trecall: {:.3f}+-{:.3f}\tPrecision {:.3f}+-{:.3f}\t'.format(
    roc_auc,
    np.mean(accuracy),
    np.std(accuracy),
    np.mean(recall),
    np.std(recall),
    np.mean(precision),
    np.std(precision))+'\tbest_distance:{:.3f}\t'.format(np.mean(best_distances))
)
