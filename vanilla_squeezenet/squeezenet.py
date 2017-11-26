import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes,
                 residual=False):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        # squeeze
        self.squeeze_bn = nn.BatchNorm2d(inplanes)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv2d(
            inplanes, squeeze_planes, kernel_size=1, bias=False
        )
        # expand
        self.expand1x1_bn = nn.BatchNorm2d(squeeze_planes)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(
            squeeze_planes, expand1x1_planes, kernel_size=1, bias=False
        )
        # expand
        self.expand3x3_bn = nn.BatchNorm2d(squeeze_planes)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1, bias=False
        )
        self.residual = residual
        if self.residual:
            assert (inplanes == (expand1x1_planes + expand3x3_planes))

    def forward(self, x):
        residual = x
        x = self.squeeze(self.squeeze_activation(self.squeeze_bn(x)))
        out = torch.cat([
            self.expand1x1(self.expand1x1_activation(self.expand1x1_bn(x))),
            self.expand3x3(self.expand3x3_activation(self.expand3x3_bn(x)))
        ], 1)

        if self.residual:
            return out + residual
        else:
            return out


class SqueezeNet(nn.Module):

    def __init__(self, num_classes=200):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64, residual=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128, residual=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192, residual=True),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256, residual=True)
        )

        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(7)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)
