import os
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import models

os.environ['TORCH_HOME'] = './SVR/ckpt/models'

class ImnetDecoder(nn.Module):
    def __init__(self):
        super(ImnetDecoder, self).__init__()
        self.global_feat_dim = 128
        self.latent_feat_dim = 32

        self.relu = nn.ReLU()

        self.l1 = nn.Linear(self.global_feat_dim+3, self.latent_feat_dim*16)
        self.l2 = nn.Linear(self.latent_feat_dim*16+3, self.latent_feat_dim*8)
        self.l3 = nn.Linear(self.latent_feat_dim*8+3, self.latent_feat_dim*4)
        self.l4 = nn.Linear(self.latent_feat_dim*4+3, self.latent_feat_dim*2)
        self.l5 = nn.Linear(self.latent_feat_dim*2+3, self.latent_feat_dim)
        self.l6 = nn.Linear(self.latent_feat_dim, self.latent_feat_dim)
        self.l7 = nn.Linear(self.latent_feat_dim, 1)

    def forward(self, globalfeat, points):
        feature = globalfeat.unsqueeze(1)
        feature = feature.repeat((1,points.size(1),1))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l1(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l2(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l3(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l4(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l5(feature))
        feature = self.relu(self.l6(feature))
        feature = self.l7(feature)
        feature = torch.reshape(feature, (points.size(0), -1, 1))
        return feature
class ResEncoder(nn.Module):
    def __init__(self, res_dir):
        super(ResEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4 

        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        self.fc1 = nn.Linear(1000,128)

    def forward(self, input_view):
        feat0 = self.relu(self.bn1(self.conv1(input_view)))
        x = self.maxpool(feat0)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        x = self.avgpool(feat4)
        x = torch.flatten(x, 1)
        featvec = self.fc(x)
        featvec = self.fc1(featvec)
        featmap_list = [feat0, feat1, feat2, feat3, feat4]
        return featvec, featmap_list

class ResDecoder(nn.Module):
    def __init__(self):
        super(ResDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.channel = 64

        self.up_conv5 = nn.Conv2d(self.channel, self.channel, (1, 1))
        self.up_conv4 = nn.Conv2d(self.channel, self.channel, (1, 1))
        self.up_conv3 = nn.Conv2d(self.channel, self.channel, (1, 1))
        self.up_conv2 = nn.Conv2d(self.channel, self.channel, (1, 1))
        self.up_conv1 = nn.Conv2d(self.channel, self.channel, (1, 1))
        self.up_conv0 = nn.Conv2d(self.channel, self.channel, (1, 1))

        self.c5_conv = nn.Conv2d(512, self.channel, (1, 1))
        self.c4_conv = nn.Conv2d(256, self.channel, (1, 1))
        self.c3_conv = nn.Conv2d(128, self.channel, (1, 1))
        self.c2_conv = nn.Conv2d(64, self.channel, (1, 1))
        self.c1_conv = nn.Conv2d(64, self.channel, (1, 1))

        self.p0_conv = nn.Conv2d(self.channel, self.channel, (3, 3), padding=1)
        self.pred_disp = nn.Conv2d(self.channel, 2, (1, 1), padding=0)
        self.relu = nn.ReLU()

    def forward(self, featmap_list):
        [feat0, feat1, feat2, feat3, feat4] = featmap_list
        p5 = self.relu(self.c5_conv(feat4))
        p4 = self.up_conv5(self.upsample(p5)) + self.relu(self.c4_conv(feat3))
        p3 = self.up_conv4(self.upsample(p4)) + self.relu(self.c3_conv(feat2))
        p2 = self.up_conv3(self.upsample(p3)) + self.relu(self.c2_conv(feat1))
        p1 = self.up_conv2(self.upsample(p2)) + self.relu(self.c1_conv(feat0))
        p0 = self.relu(self.p0_conv(p1))
        output_disp = self.pred_disp(p0)
        return output_disp