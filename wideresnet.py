# References
# https://github.com/szagoruyko/wide-residual-networks
# https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/networks/wideresnet.py
# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html#wide_resnet50_2
# https://github.com/tensorflow/models/blob/e356598a5b79a768942168b10d9c1acaa923bdb4/research/autoaugment/wrn.py
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class WideResNet(nn.Module):
    def __init__(self, depth, width, dropout_rate, n_classes=10):
        super(WideResNet, self).__init__()

        assert (depth-4)%6 == 0, 'Wide-resnet depth should be 6n+4'
        n = (depth-4) // 6
        widths = [int(v * width) for v in (16, 32, 64)]

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)

        self.layer0 = self.add_layer(
            WideResNetBlock, 16, widths[0], n, dropout_rate, stride=1,
            activate_before_residual=True)
        self.layer1 = self.add_layer(
            WideResNetBlock, widths[0], widths[1], n, dropout_rate, stride=2)
        self.layer2 = self.add_layer(
            WideResNetBlock, widths[1], widths[2], n, dropout_rate, stride=2)

        self.link0 = Linker(16, widths[0], 1, strides=1)
        self.link1 = Linker(widths[0], widths[1], 2, strides=2)
        self.link2 = Linker(widths[1], widths[2], 2, strides=2)
        self.link_final = Linker(16, widths[-1], 4, strides=4)

        self.bn = nn.BatchNorm2d(widths[-1])
        self.fc = nn.Linear(widths[-1], n_classes)

        init_parameters(self.modules())

    def add_layer(self, block, 
                  inplanes, outplanes, 
                  n_blocks, dropout_rate, stride,
                  activate_before_residual=False):
        layers = [block(inplanes, outplanes, dropout_rate, stride,
                        activate_before_residual)]

        for _ in range(1, n_blocks):
            layers.append(block(outplanes, outplanes, dropout_rate, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        first_x = x

        x = self.layer0(x) + self.link0(x)
        x = self.layer1(x) + self.link1(x)
        x = self.layer2(x) + self.link2(x)

        x += self.link_final(first_x)

        x = F.relu(self.bn(x))
        x = F.avg_pool2d(x, 8)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class WideResNetBlock(nn.Module):  
    def __init__(self, inplanes, outplanes, dropout_rate, stride,
                 activate_before_residual=False):
        super(WideResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv1 = nn.Conv2d(inplanes, outplanes, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(outplanes, outplanes, 3, stride=1, padding=1)

        if inplanes == outplanes and stride == 1:
            self.link = None
        else:
            self.link = Linker(inplanes, outplanes, stride, strides=stride)
        self.activate_before_residual = activate_before_residual
        
        init_parameters(self.modules())

    def forward(self, x):
        out0 = self.bn1(x)
        out0 = F.relu(out0, inplace=True)

        if self.activate_before_residual:
            x = out0

        out = self.conv1(out0)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.link is not None:
            return out + self.link(out0)
        else:
            return out + x


class Linker(nn.Module):  
    def __init__(self, inplanes, outplanes, kernel_size, strides):
        super(Linker, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, strides)
        self.pad = (0, 0, 0, 0) + ((outplanes-inplanes)//2,) * 2

    def forward(self, x):
        x = self.avgpool(x)
        x = F.pad(x, self.pad, 'constant', 0)
        return x


def init_parameters(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = WideResNet(28, 10, 0.3, 10)
    x = torch.zeros((32, 3, 32, 32))
    print(model(x).size())
 
