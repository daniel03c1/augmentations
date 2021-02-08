import torch
import torchvision
import torchvision.transforms.functional as TF


class EfficientCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(EfficientCIFAR10, self).__init__(*args, **kwargs)
        self.preprocess_data()

    def preprocess_data(self):
        h, w, c = self.data[0].shape
        n_samples = len(self.data)
        new_data = torch.zeros((n_samples, c, h, w))

        for i in range(n_samples):
            new_data[i] = TF.to_tensor(self.data[i])
        
        del self.data
        self.data = new_data

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

