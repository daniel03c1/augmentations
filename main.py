import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

from adv_autoaug_trainer import AdvAutoaugTrainer
from agents import PPOAgent
from agents_aug import newSGC_aug
from augments import RandAugment
from dataloader import EfficientCIFAR10
from models import *
from transforms import transforms as bag_of_ops
from wideresnet import WideResNet


def main(config, **kwargs):
    ''' DATASETS '''
    # transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            # RandAugment(bag_of_ops, 2, 14/30),
            # transforms.RandomErasing(p=1, scale=(0.25, 0.25), ratio=(1., 1.)),
        ]),
        'val': transforms.Compose([]),
    }

    # datasets & dataloaders
    dataloaders = {}
    PATH = '/datasets/datasets/cifar'
    for mode in ['train', 'val']:
        dataloaders[mode] = EfficientCIFAR10(PATH,
                                             train=mode == 'train',
                                             transform=data_transforms[mode])
        dataloaders[mode] = torch.utils.data.DataLoader(
            dataloaders[mode],
            batch_size=128,
            shuffle=mode=='train',
            drop_last=True,
            num_workers=12)

    ''' TRAINING '''
    # model
    model = WideResNet(28, 10, 0.3, n_classes=10)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          nesterov=True,
                          weight_decay=5e-4)

    normalize = transforms.Normalize(
        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    # RL
    c = SGC(bag_of_ops, op_layers=2)
    c_optimizer = optim.Adam(c.parameters(), lr=0.035)
    ppo = PPOAgent(c, name=f'{config.name}_ppo.pt', 
                   grad_norm=0.01,
                   batch_size=config.M, 
                   # ent_coef=1e-4, 
                   augmentation=None, # newSGC_aug,
                   device=torch.device('cpu'))

    trainer = AdvAutoaugTrainer(model=model,
                                optimizer=optimizer,
                                criterion=criterion,
                                name=config.name,
                                bag_of_ops=bag_of_ops,
                                rl_n_steps=12, 
                                M=M, 
                                normalize=normalize,
                                rl_agent=ppo)

    print(bag_of_ops.ops)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[60,120,160],
                                                     gamma=0.2)
    trainer.fit(dataloaders['train'], 
                dataloaders['val'], 
                n_epochs=config.epochs,
                scheduler=scheduler)


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str, required=True)
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--M', type=int, default=8)
    config = args.parse_args()

    main(config)

