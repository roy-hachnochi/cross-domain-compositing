import torch, ChamferDistancePytorch.chamfer3D.dist_chamfer_3D, ChamferDistancePytorch.fscore as fscore
import ChamferDistancePytorch.chamfer3D as chamfer3D
import argparse
import os
from plyfile import PlyElement, PlyData
import numpy as np
import tqdm
import json
import pickle
import trimesh
import scipy
import SVR.utils as utils


def get_args():
    """Create arguments."""
    parser = argparse.ArgumentParser(prog = 'evaluation program',
                                    description = 'evaluate CD for SVR pointclouds')
    parser.add_argument('--cat_id', type = str, default = '04256520', help = 'id of ShapeNet category to be tested')
    parser.add_argument('--input_dir', type = str, help = 'dir to tested pointcloud')
    parser.add_argument('--gt_dir', type = str, help = 'dir to the grount truth pointcloud')
    parser.add_argument('--tt_split', type = str, help = 'dir to the train test split list')
    parser.add_argument('--num_points',type = int, default = 20000, help = 'number of points used to evaluate 3D entities')
    args = parser.parse_args()
    return args

def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    return vertices

def compute_cd_iou(args, test_iou = True):
    """Compute L1 Chamfer distance and 3D IoU between reference and test 3D entities."""
    # Number of sample points
    num_points = args.num_points

    # Directory of reference and source pointclouds
    test_dir = os.path.join(args.input_dir, args.cat_id)
    ref_dir = os.path.join(args.gt_dir, args.cat_id)

    # Train test split files
    split_file = os.path.join(args.tt_split, args.cat_id+"_test.lst")
    with open(split_file, 'r') as f:
        models = f.read().split('\n')
    

    print(f"{len(models)} models are being tested of category {args.cat_id}")

    all_eval = {}
    chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    for m in tqdm.tqdm(models[:-1]):
        # Determine chamfer distance
        ref_pcd = torch.tensor(np.load(os.path.join(ref_dir,m,'pointcloud.npz'))['points']
                    .astype(np.float32)).unsqueeze(dim=0).cuda()
        ref_indices = np.random.choice(np.array(ref_pcd.shape[1]),num_points)
        ref_pcd = ref_pcd[:, ref_indices, :]
        test_pcd = torch.tensor(load_pointcloud(os.path.join(test_dir,m+'.ply'))).unsqueeze(dim=0).cuda()
        test_points = min(num_points,test_pcd.shape[1])
        test_indices = np.random.choice(np.array(test_pcd.shape[1]), test_points)
        assert(test_pcd.shape[1] >= test_points)
        test_pcd = test_pcd[:,test_indices,:]
        dist1, dist2, idx1, idx2 = chamLoss(ref_pcd, test_pcd)
        cd_l1 = (dist1**.5).mean(axis = -1) + (dist2**.5).mean(axis = -1)
        f_score, precision, recall = fscore.fscore(dist1, dist2)
 
        # Determine 3D IoU, users can choose not to evaluate 3D IoU by setting test_iou as False
        if test_iou:
            test_mesh = trimesh.load(os.path.join(test_dir,m+'.ply'))
            points = np.load(os.path.join(ref_dir,m,'points.npz'))['points'].astype(np.float32)
            rand_indices = np.random.choice(points.shape[0], test_points)
            occupancies_gt = np.unpackbits(np.load(os.path.join(ref_dir,m,'points.npz'))['occupancies']).astype(bool)
            points = points[rand_indices,:]
            occupancies_gt = occupancies_gt[rand_indices]
            occupancies_test = test_mesh.contains(points)
            intersection = occupancies_gt & occupancies_test
            union = occupancies_gt | occupancies_test
            iou = sum(intersection) / sum(union)
        else:
            iou = 0
        
        # Write evaluation results to a dictionary
        all_eval[m] = {'cd_l1':np.array(cd_l1.cpu()),'f_score':np.array(f_score.cpu()),
                        'precision':np.array(precision.cpu()),'recall':np.array(recall.cpu()), 'iou':iou, 
                        'dist1': np.array(dist1.cpu()), 'dist2':np.array(dist2.cpu())}
    return all_eval


def as_mesh(scene_or_mesh):
    """ Forked from: https://github.com/mikedh/trimesh/issues/507
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def rot_z(angle):
    """Rotation matrix on z axis."""
    temp = np.eye(4)
    mat = np.array([[np.cos(angle), - np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0 , 0, 1]])
    temp[:3,:3] = mat
    return temp

def rot_x(angle):
    """Roatation matrix on x axis."""
    temp = np.eye(4)
    mat = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0 , np.sin(angle), np.cos(angle)]])
    temp[:3,:3] = mat
    return temp

def rot_y(angle):
    """Roatation matrix on y axis."""
    temp = np.eye(4)
    mat = np.array([[np.cos(angle), 0, np.sin(angle)],
                   [0, 1, 0],
                   [-np.sin(angle),0, np.cos(angle)]])
    temp[:3,:3] = mat
    return temp


def eval_folder(config, folder_name, test_iou):
    """Evaluate L1 Chamfer distance and 3D IoU on test pointcloud."""
    print("evaling ", folder_name)
    # Parse arguments
    args = get_args()
    args.input_dir = './SVR/ckpt/outputs/' + config.model_folder_name + folder_name
    args.gt_dir = './SVR/data/ShapeNet'
    args.tt_split = './SVR/data/train_test_split'
    all_eval = compute_cd_iou(args, test_iou=test_iou)

    # Save the evaluation results
    with open("./SVR/result/" + folder_name + "_eval.pkl", "wb") as outfile:
        pickle.dump(all_eval, outfile)
    
    # Print L1 Chamfer distance
    cd = 0
    for key in all_eval:
        cd += all_eval[key]['cd_l1']
    cd /= len(all_eval)
    print(cd)


if __name__ == "__main__":
    config = utils.get_args()
    # NOTE: This is to be edited by the user depending on which folder they are evaluating over
    folder_name = "sofa_inpaint_050_best_model_401"
    eval_folder(config, folder_name, test_iou=True)