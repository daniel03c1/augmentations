import torch
import torchvision


class RandAugment(torch.nn.Module):
    def __init__(self, bag_of_ops, n, m):
        super(RandAugment, self).__init__()
        self.n = n
        self.trans = torchvision.transforms.RandomChoice(
            [op(m) for op in bag_of_ops.ops])

    def __call__(self, img):
        for i in range(self.n):
            img = self.trans(img)
        return img


if __name__ == '__main__':
    from transforms import transforms as bag_of_ops

    ra = RandAugment(bag_of_ops, 2, 14/30)
    xs = torch.rand((8, 3, 32, 32))
    print(ra(xs).size())

