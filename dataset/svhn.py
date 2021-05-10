import torch
from torchvision import datasets, transforms

import params


def get_svhn(train):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(params.image_size),
                                      transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

   
   ])


    # dataset and data loader
    svhn_dataset = datasets.SVHN(root=params.svhn_dataset_root,
                                   split='train' if train else 'test',
                                   transform=pre_process,
                                   download=True)

    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        drop_last=True)

    return svhn_data_loader