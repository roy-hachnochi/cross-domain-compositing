import os
from PIL import Image
import numpy as np
import torch
import SVR.utils as utils
from SVR.models.D2IM import D2IM_Net
from SVR.models.CamNet import CamNet
from SVR.datasets.SDFDataset import SDFDataset as Dataset
import math


def test_external_sdf(config, image_folder, save_path):
    SDFmodel = D2IM_Net(config)
    CAMmodel = CamNet(config)
    SDFmodel.cuda()
    CAMmodel.cuda()
    CAM_pretrain_fn = config.model_dir+f'/{config.model_folder_name}/{config.cam_model_name}'
    SDF_pretrain_fn = config.model_dir+f'/{config.model_folder_name}/{config.sdf_model_name}'
    _, CAMmodel, _, _ = utils.load_checkpoint(CAM_pretrain_fn, CAMmodel, None)
    epoch, SDFmodel, _, _ = utils.load_checkpoint(SDF_pretrain_fn, SDFmodel, None)
    CAMmodel.eval()
    SDFmodel.eval()
    with torch.no_grad():
        gridworldcoords = utils.sample_grid_points(config.mcube_znum, config.mcube_znum, config.mcube_znum) # sampled gridpoints under camera's view
        gridworldcoords = torch.tensor(gridworldcoords).float().view(-1,3).unsqueeze(0)
        if(config.cuda):
            gridworldcoords = gridworldcoords.cuda()

        
        all_image = os.listdir(image_folder)
        dataset = Dataset(config, 'test')
        for image in all_image:
            image_path = os.path.join(image_folder, image)
            image_name = image.split('.')[0]
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok =True)

            rgb_image = torch.tensor(np.array(Image.open(image_path)))
            rgb_image = torch.tensor(rgb_image).float()/255.
            rgb_image = rgb_image.permute(2,0,1)
            rgb_image = rgb_image.cuda()
            rgb_image = rgb_image.unsqueeze(0)
            
            # predict the camera
            pred_RT = CAMmodel.test(rgb_image)
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
                pred_gtvalues= SDFmodel.inference_sdf_with_detail(points_batch, rgb_image, pred_transmat)
                pred_value_list.append(pred_gtvalues)
            pred_values = torch.cat(pred_value_list, dim=1)

            # reshape the implicit field
            pred_values = pred_values.view(config.mcube_znum, config.mcube_znum,config.mcube_znum)
            pred_values = pred_values.cpu().detach().numpy()
            pred_values = pred_values/10.0

            # extract the iso-surface
            if(not os.path.exists(save_path)):
                os.makedirs(save_path)
            ply_fname = os.path.join(save_path, image_name + '.ply')
            utils.render_implicits(ply_fname, pred_values)

def main():
    config = utils.get_args()
    image_folder = "./SVR/data/sofa_sample/sofa_sample_processed"
    save_path = f"./SVR/result/sofa_sample/{config.model_folder_name}/ply_files"
    os.makedirs(save_path, exist_ok=True)
    test_external_sdf(config, image_folder, save_path)

if __name__ == "__main__":
    main()