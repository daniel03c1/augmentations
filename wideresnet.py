# References
# https://github.com/szagoruyko/wide-residual-networks
# https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/networks/wideresnet.py
# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html#wide_resnet50_2
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

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=True)

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
            inplanes, outplanes, 3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            outplanes, outplanes, 3, stride=1, padding=1, bias=True)

        if inplanes == outplanes and stride == 1:
            self.link = None
        else:
            self.link = nn.Conv2d(
                inplanes, outplanes, 1, stride=stride, bias=True)
        
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
            # nn.init.constant_(m.bias, 0)
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
 
