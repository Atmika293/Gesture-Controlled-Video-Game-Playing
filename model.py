import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        in_step_layers = []
        in_step_layers.append(nn.Conv2d(3, 64, 3, padding=1))
        in_step_layers.append(nn.BatchNorm2d(64))
        in_step_layers.append(nn.ReLU(inplace=True))
        in_step_layers.append(nn.Conv2d(64, 64, 3, padding=1))

        self.downStep0 = nn.Sequential(*in_step_layers)

        short_layers = []
        short_layers.append(nn.Conv2d(3, 64, 3, padding=1))
        short_layers.append(nn.BatchNorm2d(64))
        short_layers.append(nn.ReLU(inplace=True))

        self.short = nn.Sequential(*short_layers)

        self.downDrop1 = nn.Dropout2d(p=0.2, inplace=False)

        self.downStep1 = downStep(64, 128, dropout=True)
        self.downStep2 = downStep(128, 256, dropout=True)
        self.downStep3 = downStep(256, 512, dropout=True)

        self.bridge = downStep(512, 1024, dropout=True)

        self.upBridge = upStep(1024, 512, dropout=True)

        self.upStep3 = upStep(512, 256, dropout=True)
        self.upStep2 = upStep(256, 128, dropout=True)
        self.upStep1 = upStep(128, 64, dropout=True)

        self.final_conv = nn.Conv2d(64, 1, 1)

        self.final_layer = nn.Sigmoid()	

    def forward(self, x):
        branch1 = self.downStep0(x)
        branch1 = torch.add(branch1, 1, self.short(x))
        branch1 = self.downDrop1(branch1)

        branch2 = self.downStep1(branch1)
        branch3 = self.downStep2(branch2)
        branch4 = self.downStep3(branch3)

        x = self.bridge(branch4)

        x = self.upBridge(x, branch4)

        x = self.upStep3(x, branch3)
        x = self.upStep2(x, branch2)
        x = self.upStep1(x, branch1)

        x = self.final_conv(x)

        x = self.final_layer(x)

        return x

class downStep(nn.Module):
    def __init__(self, inC, outC, dropout=False):
        super(downStep, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(inC)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inC, outC, 3, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(outC)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outC, outC, 3, padding=1)

        self.short_conv = nn.Conv2d(inC, outC, 3, stride=2, padding=1)
        self.short_bn = nn.BatchNorm2d(outC)
        self.short_relu = nn.ReLU(inplace=True)

        self.use_dropout = dropout
        if dropout:
            self.dropout_layer = nn.Dropout2d(p=0.2, inplace=True)

    def forward(self, x):
        res_x = self.batch_norm1(x)
        res_x = self.relu1(res_x)
        res_x = self.conv1(res_x)
        res_x = self.batch_norm2(res_x)
        res_x = self.relu2(res_x)
        res_x = self.conv2(res_x)

        shortcut = self.short_conv(x)
        shortcut = self.short_bn(shortcut)
        shortcut = self.short_relu(shortcut)

        x = torch.add(res_x, 1, shortcut) 

        if self.use_dropout:
            x = self.dropout_layer(x)

        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, kernel_size=2, dropout=False):
        super(upStep, self).__init__()
        self.upconv = nn.ConvTranspose2d(inC, outC, kernel_size, stride=2)
        # self.upconv = nn.Sequential(nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True), \
        #     nn.Conv2d(inC, outC, 3, padding=1), nn.ReLU(inplace=True))
        self.batch_norm1 = nn.BatchNorm2d(inC)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inC, outC, 3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(outC)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outC, outC, 3, padding=1)

        self.short_conv = nn.Conv2d(inC, outC, 3, padding=1)
        self.short_bn = nn.BatchNorm2d(outC)
        self.short_relu = nn.ReLU(inplace=True)

        self.use_dropout = dropout
        if dropout:
            self.dropout_layer = nn.Dropout2d(p=0.2, inplace=True)
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!

    def forward(self, x, branch):
        x = self.upconv(x)
        x = torch.cat((x, branch), dim=1)

        res_x = self.batch_norm1(x)
        res_x = self.relu1(res_x)
        res_x = self.conv1(res_x)
        res_x = self.batch_norm2(res_x)
        res_x = self.relu2(res_x)
        res_x = self.conv2(res_x)

        shortcut = self.short_conv(x)
        shortcut = self.short_bn(shortcut)
        shortcut = self.short_relu(shortcut)

        x = torch.add(res_x, 1, shortcut)

        if self.use_dropout:
            x = self.dropout_layer(x) 

        return x