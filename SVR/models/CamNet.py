import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

os.environ['TORCH_HOME'] = './SVR/ckpt/models'

class CamNet(nn.Module):
    def __init__(self, config):
        super(CamNet, self).__init__()
        self.config = config
        self.vgg = models.vgg16(pretrained=True)
        
        self.m1_fc1 = nn.Linear(1000, 512)
        self.m1_fc2 = nn.Linear(512, 256)
        self.m1_fc3 = nn.Linear(256, 1)
        torch.nn.init.normal_(self.m1_fc3.weight, 0.0, 0.05)
        torch.nn.init.constant_(self.m1_fc3.bias, 0.0)

        self.m2_fc1 = nn.Linear(1000, 512)
        self.m2_fc2 = nn.Linear(512, 256)
        self.m2_fc3 = nn.Linear(256, 6)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4= nn.BatchNorm1d(256)
        self.sigmoid = nn.Sigmoid()
        self.mseloss = nn.MSELoss()

        self.CAM_MAX_DIST = 1.75
        self.CAM_ROT = torch.tensor(np.asarray([[0.0, 0.0, 1.0],
                                      [1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0]], dtype=np.float32))
        self.R_camfix = torch.tensor(np.matrix(((1., 0., 0.), (0., -1., 0.), (0., 0., -1.)), dtype=np.float32))
        self.R_camfix = torch.transpose(self.R_camfix, 0, 1)
        self.zerovec = torch.zeros(2)
        self.maxveclen = torch.tensor(1e-8)
        
        if(config.cuda):
            self.CAM_ROT = self.CAM_ROT.cuda()
            self.R_camfix = self.R_camfix.cuda()
            self.zerovec = self.zerovec.cuda()
            self.maxveclen = self.maxveclen.cuda()
            
    def normalize_vector(self, v):
        v_mag = torch.sqrt(torch.sum(v.mul(v), dim=1).unsqueeze(1))
        v_mag = torch.max(v_mag, self.maxveclen.repeat(v_mag.size()))
        v = v / v_mag.repeat(1,v.size(1))
        return v

    def compute_rotation_matrix_from_ortho6d(self, pred_rotation):
        x_raw = pred_rotation[:,0:3]
        y_raw = pred_rotation[:,3:6]
        x = self.normalize_vector(x_raw) 
        z = torch.cross(x,y_raw,dim=1) 
        z = self.normalize_vector(z)
        y = torch.cross(z,x,dim=1) 
        x = x.unsqueeze(2)
        y = y.unsqueeze(2)
        z = z.unsqueeze(2)
        output = torch.cat([x,y,z], dim=2) # output: [B,3,3]
        return output

    def transform_points(self, points, transmat):
        plus = torch.ones((points.size(0),points.size(1),1)).cuda()
        homopoints = torch.cat([points,plus], dim=2)
        transformed = torch.matmul(homopoints, transmat)
        return transformed
    
    def forward(self, image, gt_transmat, samplings):
        globalfeat = self.vgg(image)

        # predict the translation
        m1 = self.bn1(self.m1_fc1(globalfeat))
        m1 = self.bn2(self.m1_fc2(m1))
        pred_translation = self.m1_fc3(m1)

        # predict the rotation
        m2 = self.bn3(self.m2_fc1(globalfeat))
        m2 = self.bn4(self.m2_fc2(m2))
        pred_rotation = self.m2_fc3(m2)
        
        # compute the transformation matrix
        cam_location_inv = torch.cat([pred_translation * self.CAM_MAX_DIST, self.zerovec.unsqueeze(0).repeat(pred_translation.size(0),1)], dim=1)
        R_obj2cam_inv = self.CAM_ROT.unsqueeze(0).repeat(pred_translation.size(0),1,1)
        R_camfix_inv = self.R_camfix.unsqueeze(0).repeat(pred_translation.size(0),1,1)
        
        cam_location_inv = cam_location_inv.unsqueeze(1)
        pred_translation_inv = cam_location_inv.matmul(R_obj2cam_inv)
        pred_translation_inv = pred_translation_inv.matmul(R_camfix_inv)
        pred_translation_inv = pred_translation_inv * (-1.0)
        
        pred_rotation_mat_inv = self.compute_rotation_matrix_from_ortho6d(pred_rotation)
        pred_transmat = torch.cat([pred_rotation_mat_inv, pred_translation_inv], dim=1)
        
        # project the points to the camera view
        gt_pos = self.transform_points(samplings, gt_transmat)
        pred_pos = self.transform_points(samplings, pred_transmat)
        loss = self.mseloss(pred_pos, gt_pos)
        
        return pred_transmat, loss

    def test(self, image):
        image = image[:,:3,:,:]
        globalfeat = self.vgg(image)

        # predict the translation
        m1 = self.bn1(self.m1_fc1(globalfeat))
        m1 = self.bn2(self.m1_fc2(m1))
        pred_translation = self.m1_fc3(m1)

        # predict the rotation
        m2 = self.bn3(self.m2_fc1(globalfeat))
        m2 = self.bn4(self.m2_fc2(m2))
        pred_rotation = self.m2_fc3(m2)
        
        # compute the transformation matrix
        cam_location_inv = torch.cat([pred_translation * self.CAM_MAX_DIST, self.zerovec.unsqueeze(0).repeat(pred_translation.size(0),1)], dim=1)
        R_obj2cam_inv = self.CAM_ROT.unsqueeze(0).repeat(pred_translation.size(0),1,1)
        R_camfix_inv = self.R_camfix.unsqueeze(0).repeat(pred_translation.size(0),1,1)
        
        cam_location_inv = cam_location_inv.unsqueeze(1)
        pred_translation_inv = cam_location_inv.matmul(R_obj2cam_inv)
        pred_translation_inv = pred_translation_inv.matmul(R_camfix_inv)
        pred_translation_inv = pred_translation_inv * (-1.0)
        
        pred_rotation_mat_inv = self.compute_rotation_matrix_from_ortho6d(pred_rotation)
        pred_transmat = torch.cat([pred_rotation_mat_inv, pred_translation_inv], dim=1)
        
        return pred_transmat

    