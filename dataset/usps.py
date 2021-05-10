import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split

import params


def get_usps(train):
    """Get usps dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(params.image_size),
                                      transforms.ToTensor(),
        # transforms.Normalize((0.5),(0.5)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))

        
        ])


    # dataset and data loader
    usps_dataset = datasets.USPS(root=params.usps_dataset_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)



    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        drop_last=True)
    return usps_data_loader