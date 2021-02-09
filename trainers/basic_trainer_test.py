import os
import unittest
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from basic_trainer import *


class BasicTrainerTest(unittest.TestCase):
    def setUp(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # dataloader
        self.dataloader = datasets.CIFAR10('/media/data1/datasets/cifar',
                                           train=False, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(self.dataloader,
                                                      batch_size=64,
                                                      shuffle=False,
                                                      num_workers=4)

        # model
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def test_classification_trainer(self):
        old_param = self.model.state_dict()

        trainer = ClassificationTrainer(self.model, self.optimizer, self.criterion, 'test.pt')
        _, old_acc = trainer.run(self.dataloader, train=False)

        # train (for efficiency, train on validation dataset)
        trainer.fit(self.dataloader, self.dataloader, 1)
        _, new_acc = trainer.run(self.dataloader, train=False)
        self.assertGreater(new_acc, old_acc)

        # reset params
        self.model.load_state_dict(old_param)


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    unittest.main()

