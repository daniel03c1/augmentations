import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

from agents import DiscretePPOAgent
from augments import RandAugment
from controllers import *
from dataloader import *
from trainers import Trainer
from discrete_transforms import transforms as bag_of_ops
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
    if config.dataset == 'cifar10':
        dataset = EfficientCIFAR10
        n_classes = 10

        normalize = transforms.Normalize(
            [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    elif config.dataset == 'cifar100':
        dataset = EfficientCIFAR100
        n_classes = 100

        normalize = transforms.Normalize(
            [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    else:
        raise ValueError('invalid dataset')

    dataloaders = {}
    for mode in ['train', 'val']:
        dataloaders[mode] = dataset(config.data_path,
                                    train=mode == 'train',
                                    transform=data_transforms[mode])
        dataloaders[mode] = torch.utils.data.DataLoader(
            dataloaders[mode],
            batch_size=config.batch_size,
            shuffle=mode=='train',
            drop_last=True,
            num_workers=6)

    ''' TRAINING '''
    # model
    model = WideResNet(28, 10, 0., n_classes)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          nesterov=True,
                          weight_decay=5e-4)

    # RL
    c = SGCvD(bag_of_ops, op_layers=2)
    c_optimizer = optim.Adam(c.parameters(), lr=0.035)
    ppo = DiscretePPOAgent(c, name=f'{config.name}_ppo.pt', 
                           mem_maxlen=config.mem_size*config.M,
                           batch_size=config.M, 
                           ent_coef=config.ent_coef,
                           device=torch.device('cpu'))

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      name=config.name,
                      bag_of_ops=bag_of_ops,
                      rl_n_steps=config.rl_steps, 
                      deprecation_rate=config.gamma,
                      M=config.M, 
                      normalize=normalize,
                      rl_agent=ppo)

    print(bag_of_ops.ops)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           config.epochs)
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
    args.add_argument('--mem_size', type=int, default=1)
    args.add_argument('--rl_steps', type=int, default=2)
    args.add_argument('--ent_coef', type=float, default=1e-5)
    args.add_argument('--gamma', type=float, default=1.)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--dataset', type=str, default='cifar100')
    args.add_argument('--data_path', type=str, 
                      default='/datasets/datasets/cifar')
    config = args.parse_args()

    main(config)

