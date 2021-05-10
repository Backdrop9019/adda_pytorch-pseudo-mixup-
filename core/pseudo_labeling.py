import torch.nn as nn
import torch
import torch.optim as optim

import params
from utils import make_cuda, save_model, LabelSmoothingCrossEntropy,mixup_data
from random import *
import sys

from torch.utils.data import Dataset,DataLoader




def pseudo_labeling(encoder, classifier,target_data_loader,threshold=0.9):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.eval()
    classifier.eval()
    
    softmax = nn.Softmax(dim=1)

    custom_data = []
    custom_label = []

    ######################
    # 2. pseudo labeling #
    ######################


    for step, (images, _) in enumerate(target_data_loader):

        # make images and labels variable
        images = make_cuda(images)
        preds = classifier(encoder(images))
        labels = torch.max(softmax(preds),1).indices
        condition = torch.max(softmax(preds),1).values >threshold
        for i in images[condition]:
            custom_data.append(i )
        for i in labels[condition]:
            custom_label.append(i)

    #######################
    # 3. make data loader #
    #######################
    
    #torch.Size([size, 3, 32, 32])
    img = torch.stack(custom_data)
    # torch.Size([size])
    label = torch.stack(custom_label)
    



    print(f'complete pseudo labeling {img.size(0)} out of total {len(target_data_loader.dataset)}')
    if img.size(0) < 1:
        print('No data exists that satisfies the threshold.')
        return
    #create custom datasets
    dataset = CustomDataset(img,label)

    pseudo_data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=params.batch_size,
    shuffle=True,
    drop_last=True)


    return pseudo_data_loader


class CustomDataset(Dataset): 
  def __init__(self,img,label):
    self.x_data = img
    self.y_data = label
  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = img[idx]
    y = label[idx]
    return x, y


    