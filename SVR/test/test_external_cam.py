"""Get camera pose on test images."""
import os
from PIL import Image
import numpy as np
import torch
import SVR.utils as utils
from plyfile import PlyElement, PlyData
from SVR.models.D2IM import D2IM_Net
from SVR.models.CamNet import CamNet
from SVR.datasets.SDFDataset import SDFDataset as Dataset
import math
from natsort import natsorted

def solve_a_e(matrix):
    """Determine yaw and row from rotation matrix"""
    cos_e = matrix[2,2]
    cos_a = matrix[1,1]
    sin_e = -matrix[2,0]
    sin_a = -matrix[0,1]
    a = np.arctan2(sin_a, cos_a)
    e = np.arctan2(sin_e, cos_e)
    if not ((abs(cos_a - np.cos(a)) < 1e-4) & (abs(sin_a - np.sin(a)) < 1e-4)):
        a = np.pi - a
    if not ((abs(cos_e - np.cos(e)) < 1e-4) & (abs(sin_e - np.sin(e)) < 1e-4)):
        e = np.pi - e
        
    return -np.degrees(a), -np.degrees(e)

def retrive_parameters_from_P(RT):
    """Get yaw, row and distance ratio from rotation matrix"""    
    R = RT[:3,:3]
    T = RT[:,3].reshape((3,1))
    CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                          [1.0, -4.371138828673793e-08, -0.0],
                          [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])
    R_cam_fix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    T = R_cam_fix @ T
    R = R_cam_fix @ R
    cam_location = -1 * CAM_ROT @ T 
    R_world2obj = (CAM_ROT @ R).T
    a ,e = solve_a_e(R_world2obj)
    dist_ratio = np.asarray((cam_location[0]/1.75))[0][0]
    return a, e, dist_ratio

def get_K(img_w = 224, img_h = 224):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    F_MM = 35.  # Focal length
    SENSOR_SIZE_MM = 32.
    PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.
    SKEW = 0.
    # Calculate intrinsic matrix.
    # 2 atan(35 / 2*32)
    scale = RESOLUTION_PCT / 100
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    return K

def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    return vertices

def load_camera_model(model_path):
    config = utils.get_args()
    CAMmodel = CamNet(config)
    CAMmodel.cuda()
    _, CAMmodel, _, _ = utils.load_checkpoint(model_path, CAMmodel, None)
    CAMmodel.eval()
    return CAMmodel

def predict_render_param(image, model):
    with torch.no_grad():
        rgb_image = torch.tensor(np.array(Image.open(image)))
        rgb_image = torch.tensor(rgb_image).float()/255.
        rgb_image = rgb_image.permute(2,0,1)
        rgb_image = rgb_image.cuda()
        rgb_image = rgb_image.unsqueeze(0)
        
        # predict the camera
        pred_RT = model.test(rgb_image)
        pred_RT = pred_RT[0]
        pred_RT = pred_RT.cpu().detach().numpy()
        pred_RT = np.transpose(pred_RT)
        a, e, dist_ratio = retrive_parameters_from_P(pred_RT)
    return a, e, dist_ratio

def test_external_pose(config, image_folder, save_path):
    cam_model_path = f"./SVR/ckpt/models/{config.model_folder_name}/{config.cam_model_name}"
    files_save_path = os.path.join(save_path, "render_data.txt")
    model = load_camera_model(cam_model_path)
    params = []
    folder_name = image_folder.split("/")[-1]
    camera_name = cam_model_path.split("/")[-1]
    print(f"Evaluating images in {folder_name} with camera model {camera_name}")
    for image in natsorted(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image)
        a, e, dist_ratio = predict_render_param(image_path, model)
        params.append([a ,e ,0, dist_ratio,35,32,1.75,0.0,0.0,0.0])
    with open(files_save_path, 'w') as fp:
        for param in params:
            fp.write("%s\n" % param)

def main():
    image_folder = "./SVR/data/sofa_sample/sofa_sample_processed" #Path to the test images
    config = utils.get_args()
    save_path = f"./SVR/result/sofa_sample/{config.model_folder_name}"
    os.makedirs(save_path, exist_ok=True)
    test_external_pose(config, image_folder, save_path)

if __name__ == "__main__":
    main()