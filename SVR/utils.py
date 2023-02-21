import os
import cv2
import mcubes
import torch
import numpy as np
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='D2IM-Net')

    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--plot_every_batch', type=int, default=10)
    parser.add_argument('--save_every_epoch', type=int, default=20)
    parser.add_argument('--test_every_epoch', type=int, default=20)
    parser.add_argument('--load_pretrain', type=bool, default=True)

    parser.add_argument('--viewnum', type=int, default=36)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--mcube_znum', type=int, default=128)
    parser.add_argument('--test_pointnum', type=int, default=100000)

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--cam_batch_size', type=int, default=16)
    parser.add_argument('--cam_lr', type=float, default=0.00005)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--sampling_mode', type=str, default='weighted')
    parser.add_argument('--exp_name', type=str, default='d2im')
    
    # Data preperation hyperparameters
    parser.add_argument('--train_augmented', type = bool, default = True, help = "Indicating if the training is over augmented images")
    parser.add_argument('--augmented_cam_views', default = [0, 1, 2, 33, 34, 35], help = "List of camera view points to augment,indexed from DISN dataset")
    parser.add_argument('--augmentation_strength', type = float, default = 0.5, help = 'Foreground condition strength, in range [0., 1.]')
    parser.add_argument('--prompt', type = str, default = "A photograph of a sofa in a living room with windows and a rug", help = "text prompt used to augment images, the longer and preciser the better/")
    parser.add_argument('--model_folder_name', default = "inpaint_050/", help = "subfolder name to save trained models under different sub directories.")
    parser.add_argument('--sdf_model_name', default = 'sofa_inpaint_050_best_model.pt.tar', help = "File name to save trained SDF model, to be changed for different training data." )
    parser.add_argument('--cam_model_name', default = 'sofa_inpaint_050_best_model_cam.pt.tar', help = "File name to save trained cam model, to be changed for different training data")

    # Data path
    parser.add_argument('--data_dir', default='./SVR/data/train_test_split/', help = 'Path to the train test split dir')
    parser.add_argument('--h5_dir', default='./SVR/data/SDF_with_gradient/', help = "Processed gradient data produced by running preprocessing scripts")
    parser.add_argument('--density_dir', default='./SVR/data/SDF_density/', help = "Processed density data produced by running preprocessing scripts")
    parser.add_argument('--cam_dir', default='./SVR/data/image/', help = "Directory containing rendering camera metadata")
    parser.add_argument('--image_dir', default='./SVR/data/image_augmented_inpaint_050/', help = 'Path to training images')
    parser.add_argument('--normal_dir', default='./SVR/data/normal_processed/', help = 'Processed image normals by running preprocessing scripts')
    
    # Save directories
    parser.add_argument('--model_dir', default='./SVR/ckpt/models', help = 'writing/reading dir of trained models')
    parser.add_argument('--output_dir', default='./SVR/ckpt/outputs', help = 'dir to save test results')
    parser.add_argument('--log', default='log.txt')
    
    # selected sofa
    testlist = [
        # {'cat_id': '04256520', 'shape_id':'7c9e1876b1643e93f9377e1922a21892', 'cam_id':35},
        # {'cat_id': '04256520', 'shape_id':'6c930734ea1ea82476d342de8af45d5', 'cam_id':35},
        # {'cat_id': '04256520', 'shape_id':'2f0f7c2f9e3b04a8f251828d7c328331', 'cam_id':35},
        # {'cat_id': '04256520', 'shape_id':'3ac6d1680c7e0ba4fb315ce917a9ec2', 'cam_id':35},
        # {'cat_id': '04256520', 'shape_id':'4f2ab57322d7a7e3df63d1c7e43c783f', 'cam_id':35},
        # {'cat_id': '04256520', 'shape_id':'4c1ac8bd57beb2a9ce59ea70152320fa', 'cam_id':35}
    ]
   

    args = parser.parse_args()
    args.testlist = testlist
    args.catlist = ['04256520']
 
    return args

def print_log(log_fname, logline):
    f = open(log_fname,'a')
    f.write(logline)
    f.write('\n')
    f.close()

def save_checkpoint(epoch, model, optimizer, bestloss, output_filename):
    state = {'epoch': epoch + 1,
         'state_dict': model.state_dict(),
         'optimizer': optimizer.state_dict(),
         'bestloss': bestloss}
    torch.save(state, output_filename)

def load_checkpoint(cp_filename, model, optimizer=None):
    checkpoint = torch.load(cp_filename)
    model.load_state_dict(checkpoint['state_dict'])
    if(optimizer is not None):
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    if('bestloss' in checkpoint.keys()):
        bestloss = checkpoint['bestloss']
    else:
        bestloss = 10000000
    return epoch, model, optimizer, bestloss

def load_model(cp_filename, model):
    checkpoint = torch.load(cp_filename)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    return epoch, model

""" sample grid points in the 3D space [-0.5,0.5]^3 """
def sample_grid_points(xnum,ynum,znum):
    gridpoints = np.zeros((xnum, ynum, znum, 3))
    for i in range(xnum):
        for j in range(ynum):
            for k in range(znum):
                gridpoints[i, j, k, :] = [i,j,k] 
    gridpoints[:,:,:,0] = (gridpoints[:,:,:,0] + 0.5)/xnum - 0.5
    gridpoints[:,:,:,1] = (gridpoints[:,:,:,1] + 0.5)/ynum - 0.5
    gridpoints[:,:,:,2] = (gridpoints[:,:,:,2] + 0.5)/znum - 0.5
    return gridpoints

""" render the occupancy field to 3 image views """
def render_grid_occupancy(fname, gridvalues, threshold=0):
    signmat = np.sign(gridvalues - threshold)
    img1 = np.clip((np.amax(signmat, axis=0)-np.amin(signmat, axis=0))*256, 0,255).astype(np.uint8)
    img2 = np.clip((np.amax(signmat, axis=1)-np.amin(signmat, axis=1))*256, 0,255).astype(np.uint8)
    img3 = np.clip((np.amax(signmat, axis=2)-np.amin(signmat, axis=2))*256, 0,255).astype(np.uint8)

    fname_without_suffix = fname[:-4]
    cv2.imwrite(fname_without_suffix+'_1.png',img1)
    cv2.imwrite(fname_without_suffix+'_2.png',img2)
    cv2.imwrite(fname_without_suffix+'_3.png',img3)

""" marching cube """
def render_implicits(fname, gridvalues, threshold=0):
    vertices, triangles = mcubes.marching_cubes(-1.0*gridvalues, threshold)
    vertices[:,0] = ((vertices[:,0] + 0.5)/gridvalues.shape[0] - 0.5)
    vertices[:,1] = ((vertices[:,1] + 0.5)/gridvalues.shape[1] - 0.5)
    vertices[:,2] = ((vertices[:,2] + 0.5)/gridvalues.shape[2] - 0.5)    
    write_ply(fname, vertices, triangles)

def write_obj(fname, vertices, triangles):
    fout = open(fname, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(int(triangles[ii,0])+1)+" "+str(int(triangles[ii,1])+1)+" "+str(int(triangles[ii,2])+1)+"\n")
    fout.close()

def write_ply(fname, vertices, triangles):
	fout = open(fname, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("element face "+str(len(triangles))+"\n")
	fout.write("property list uchar int vertex_index\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
	fout.close()

