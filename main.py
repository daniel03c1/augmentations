import copy
import time
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from trainers import ClassificationTrainer
from dataloader import EfficientCIFAR10


def main(**kwargs):
    # transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # datasets & dataloaders
    dataloaders = {}
    for mode in ['train', 'val']:
        dataloaders[mode] = EfficientCIFAR10('/media/data1/datasets/cifar', 
                                             train=mode == 'train',
                                             transform=data_transforms[mode])
        dataloaders[mode] = torch.utils.data.DataLoader(
            dataloaders[mode],
            batch_size=32,
            shuffle=mode=='train',
            num_workers=4)

    # model
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    trainer = ClassificationTrainer(model, optimizer, criterion, 'test.pt')
    trainer.fit(dataloaders['train'], dataloaders['val'], 2)


if __name__ == '__main__':
    main()

