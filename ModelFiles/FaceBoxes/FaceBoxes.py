import torch
import torch.nn as nn
import torch.nn.functional as F


class CReLUM(nn.Module):
    def __init__(self):
        super(CReLUM, self).__init__()

    def forward(self, x):
        return F.relu(torch.cat((x, -x), 1))


CRelu = CReLUM()


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1down, n1x1up, n3x3):
        super(Inception, self).__init__()

        self.conv1 = BasicConv2d(in_planes, n1x1down, kernel_size=1)

        self.pool2_1 = nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True)
        self.conv2_2 = BasicConv2d(in_planes, n1x1down, kernel_size=1)

        self.conv3_1 = BasicConv2d(in_planes, n1x1up, kernel_size=1)
        self.conv3_2 = BasicConv2d(n1x1up, n3x3, kernel_size=3, padding=1)

        self.conv4_1 = BasicConv2d(in_planes, n1x1up, kernel_size=1)
        self.conv4_2 = BasicConv2d(n1x1up, n3x3, kernel_size=3, padding=1)
        self.conv4_3 = BasicConv2d(n3x3, n3x3, kernel_size=3, padding=1)

    def forward(self, x):
        y1 = self.conv1(x)

        y2 = self.pool2_1(x)
        y2 = self.conv2_2(y2)

        y3 = self.conv3_1(x)
        y3 = self.conv3_2(y3)

        y4 = self.conv4_1(x)
        y4 = self.conv4_2(y4)
        y4 = self.conv4_3(y4)

        return torch.cat([y1, y2, y3, y4], 1)


anchors = (21, 1, 1)


class FaceBoxes(nn.Module):
    def __init__(self):
        super(FaceBoxes, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(16, eps=0.001)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001)
        self.inception1 = Inception(128, 32, 16, 32)
        self.inception2 = Inception(128, 32, 16, 32)
        self.inception3 = Inception(128, 32, 16, 32)
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.score_conv1 = nn.Conv2d(
            128, 2 * anchors[0], kernel_size=3, stride=1, padding=1)
        self.bbox_conv1 = nn.Conv2d(
            128, 4 * anchors[0], kernel_size=3, stride=1, padding=1)
        self.score_conv2 = nn.Conv2d(
            256, 2 * anchors[1], kernel_size=3, stride=1, padding=1)
        self.bbox_conv2 = nn.Conv2d(
            256, 4 * anchors[1], kernel_size=3, stride=1, padding=1)
        self.score_conv3 = nn.Conv2d(
            256, 2 * anchors[2], kernel_size=3, stride=1, padding=1)
        self.bbox_conv3 = nn.Conv2d(
            256, 4 * anchors[2], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(CRelu(x), kernel_size=3, stride=2, ceil_mode=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(CRelu(x), kernel_size=3, stride=2, ceil_mode=True)

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)

        score1 = self.score_conv1(x)
        bbox1 = self.bbox_conv1(x)

        x = F.relu(self.conv3_1(x), inplace=True)
        x = F.relu(self.conv3_2(x), inplace=True)

        score2 = self.score_conv2(x)
        bbox2 = self.bbox_conv2(x)

        x = F.relu(self.conv4_1(x), inplace=True)
        x = F.relu(self.conv4_2(x), inplace=True)

        score3 = self.score_conv3(x)
        bbox3 = self.bbox_conv3(x)

        scorelist = list()
        bboxlist = list()
        scorelist.append(score1.permute(0, 2, 3, 1).contiguous())
        scorelist.append(score2.permute(0, 2, 3, 1).contiguous())
        scorelist.append(score3.permute(0, 2, 3, 1).contiguous())
        bboxlist.append(bbox1.permute(0, 2, 3, 1).contiguous())
        bboxlist.append(bbox2.permute(0, 2, 3, 1).contiguous())
        bboxlist.append(bbox3.permute(0, 2, 3, 1).contiguous())
        pscore = torch.cat([o.view(o.size(0), -1) for o in scorelist], 1)
        pbbox = torch.cat([o.view(o.size(0), -1) for o in bboxlist], 1)

        return pscore, pbbox
