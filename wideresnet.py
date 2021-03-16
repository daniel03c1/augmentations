# References
# https://github.com/szagoruyko/wide-residual-networks
# https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/networks/wideresnet.py
# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html#wide_resnet50_2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


'''
class WideResNet(nn.Module):
    def __init__(self, depth, width, dropout_rate, n_classes=10):
        super(WideResNet, self).__init__()

        assert (depth-4)%6 == 0, 'Wide-resnet depth should be 6n+4'
        n = (depth-4) // 6
        widths = [int(v * width) for v in (16, 32, 64)]

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)

        self.layer0 = self.add_layer(
            WideResNetBlock, 16, widths[0], n, dropout_rate, stride=1)
        self.layer1 = self.add_layer(
            WideResNetBlock, widths[0], widths[1], n, dropout_rate, stride=2)
        self.layer2 = self.add_layer(
            WideResNetBlock, widths[1], widths[2], n, dropout_rate, stride=2)

        self.bn = nn.BatchNorm2d(widths[-1])
        self.fc = nn.Linear(widths[-1], n_classes)

        init_parameters(self.modules())

    def add_layer(self, block, 
                  inplanes, outplanes, 
                  n_blocks, dropout_rate, stride):
        layers = [block(inplanes, outplanes, dropout_rate, stride)]

        for _ in range(1, n_blocks):
            layers.append(block(outplanes, outplanes, dropout_rate, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = F.relu(self.bn(x))
        x = F.avg_pool2d(x, 8)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class WideResNetBlock(nn.Module):  
    def __init__(self, inplanes, outplanes, dropout_rate, stride):
        super(WideResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv1 = nn.Conv2d(
            inplanes, outplanes, 3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            outplanes, outplanes, 3, stride=1, padding=1, bias=False)

        if inplanes == outplanes and stride == 1:
            self.link = None
        else:
            self.link = nn.Conv2d(
                inplanes, outplanes, 1, stride=stride, bias=False)
        
        init_parameters(self.modules())

    def forward(self, x):
        out0 = self.bn1(x)
        out0 = F.relu(out0, inplace=True)

        out = self.conv1(out0)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.link is not None:
            return out + self.link(out0)
        else:
            return out + x


def init_parameters(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
'''
# import torch
# import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


_bn_momentum = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=_bn_momentum)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=_bn_momentum)
        self.linear = nn.Linear(nStages[3], num_classes)

        # self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


if __name__ == '__main__':
    model = WideResNet(28, 10, 0.3, 10)
    x = torch.zeros((32, 3, 32, 32))
    print(model(x).size())
 
