from torchvision import transforms, datasets
import torch
import params

def get_custom(train):

    pre_process = transforms.Compose([transforms.Resize(params.image_size),
                                      transforms.ToTensor(),
  #  transforms.Normalize((0.5),(0.5)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    custom_dataset = datasets.ImageFolder(
       root = params.custom_dataset_root,
            transform = pre_process,
    )

    custom_data_loader = torch.utils.data.DataLoader(
        custom_dataset,
        batch_size=params.batch_size,
          shuffle=True,
        drop_last=True

    )

    return custom_data_loader