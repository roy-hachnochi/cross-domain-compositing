import os, math, cv2
import numpy as np
import random
import torch

import SVR.utils as utils
from SVR.models.D2IM import D2IM_Net
from SVR.models.CamNet import CamNet

from SVR.datasets.SDFDataset import SDFDataset as Dataset

""" test one data and output the extracted iso-surface """
def test_one_without_gttransmat(CAMmodel, SDFmodel, dataset, gridworldcoords, cat_id, shape_id, cam_id, output_dir, config):
    rgba_image, _, _ = dataset.get_testdata(cat_id, shape_id, cam_id)
    if((rgba_image is None)):
        return 
    if(config.cuda):
        rgba_image = rgba_image.cuda()
    rgba_image = rgba_image.unsqueeze(0)
    
    # predict the camera
    pred_RT = CAMmodel.test(rgba_image)
    pred_RT = pred_RT[0]
    pred_RT = pred_RT.cpu().detach().numpy()
    pred_RT = np.transpose(pred_RT)
    pred_transmat = dataset.process_transmat(pred_RT)
    pred_transmat = torch.tensor(pred_transmat).float()
    pred_transmat = pred_transmat.cuda()
    pred_transmat = pred_transmat.unsqueeze(0)
    
    # test
    test_pointnum = config.test_pointnum
    times = math.ceil(gridworldcoords.size(1)/test_pointnum)
    pred_value_list = []
    for i in range(times):
        endind = min((i+1)*test_pointnum, gridworldcoords.size(1))
        points_batch = gridworldcoords[:,i*test_pointnum:endind,:]
        pred_gtvalues= SDFmodel.inference_sdf_with_detail(points_batch, rgba_image, pred_transmat)
        pred_value_list.append(pred_gtvalues)
    pred_values = torch.cat(pred_value_list, dim=1)
    
    # reshape the implicit field
    pred_values = pred_values.view(config.mcube_znum, config.mcube_znum,config.mcube_znum)
    pred_values = pred_values.cpu().detach().numpy()
    pred_values = pred_values/10.0

    # extract the iso-surface
    if(not os.path.exists(output_dir + '/' + cat_id)):
        os.makedirs(output_dir + '/' + cat_id)
    ply_fname = output_dir + '/' + cat_id + '/' + shape_id + '.ply'
    utils.render_implicits(ply_fname, pred_values)
    #cv2.imwrite("image.png",(pred_disp+0.5)*128.0+100) 

def test_all(config):
    if(config.cuda):
        torch.cuda.set_device(config.gpu)
        torch.backends.cudnn.benchmark = True

    SDFmodel = D2IM_Net(config)
    CAMmodel = CamNet(config)
    if(config.cuda):
        SDFmodel.cuda()
        CAMmodel.cuda()
    
    testset = Dataset(config, 'test')
    test_iter = torch.utils.data.DataLoader(testset, batch_size=config.train_batch_size, shuffle=False)
    
    epoch = 0
    CAM_pretrain_fn = config.model_dir +'/' + config.model_folder_name + config.cam_model_name
    SDF_pretrain_fn = config.model_dir+ '/' + config.model_folder_name + config.sdf_model_name

    if(config.load_pretrain and os.path.exists(CAM_pretrain_fn) and os.path.exists(SDF_pretrain_fn)):
        _, CAMmodel, _, _ = utils.load_checkpoint(CAM_pretrain_fn, CAMmodel, None)
        epoch, SDFmodel, _, _ = utils.load_checkpoint(SDF_pretrain_fn, SDFmodel, None)
    else:
        print('pre-trained model doesnt exist!')
    
    
    output_dir = config.output_dir+ '/' + config.model_folder_name + config.sdf_model_name.split('.')[0]+'_'+str(epoch+1)
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    SDFmodel.eval()
    CAMmodel.eval()
    with torch.no_grad():
        gridworldcoords = utils.sample_grid_points(config.mcube_znum, config.mcube_znum, config.mcube_znum) # sampled gridpoints under camera's view
        gridworldcoords = torch.tensor(gridworldcoords).float().view(-1,3).unsqueeze(0)
        if(config.cuda):
            gridworldcoords = gridworldcoords.cuda()
        index = 0
        for data in testset.datalist:
            cat_id = data['cat_id']
            shape_id = data['shape_id']
            # select the view with most details            
            cam_id = 33
         
            test_one_without_gttransmat(CAMmodel, SDFmodel, testset, gridworldcoords, cat_id, shape_id, cam_id, output_dir, config)  

if __name__ == "__main__":
    config = utils.get_args()
    test_all(config)