import os
import cv2
import h5py
import random
import numpy as np

import torch
from torch.utils import data
import SVR.datasets.DISN_CamThing as CamUtils
rot_mat = CamUtils.get_rotate_matrix(-np.pi/2)

class SDFDataset(data.Dataset):
    def __init__(self, config, status):
        self.catlist = config.catlist
        self.viewnum = config.viewnum
        self.sampling_mode = config.sampling_mode
        self.config = config
        self.num_points = 2048
        self.datalist = []
        self.datasize = 0

        # only load the training set or test set
        if(not (status=='test' or status=='train')):
            return

        # read the dataset
        datalist = []
        for cat_id in self.catlist:
            filename = config.data_dir + cat_id + '_' + status + '.lst'
            shape_ids = self.read_shape_ids_from_file(filename)
            for shape_id in shape_ids:
                if config.train_augmented:
                    rgb_fn = config.image_dir + cat_id + '/' + shape_id + "/"
                else:
                    rgb_fn = config.image_dir + cat_id + '/' + shape_id + "/easy/"
                normal_fn = config.normal_dir + cat_id + '_easy/' + shape_id + '/'
                cam_fn = config.cam_dir + cat_id + '/' + shape_id + '/easy'+'/rendering_metadata.txt'
                h5_fn = config.h5_dir + cat_id + '/' + shape_id + '/data.h5'
                density_fn = config.density_dir + cat_id + '/' + shape_id + '/density.h5'
                if(os.path.exists(h5_fn) and os.path.exists(cam_fn) and os.path.exists(rgb_fn)):
                    data = {'rgba_dir':rgb_fn, 'normal_dir':normal_fn, 
                    'h5_fn':h5_fn, 'cam_fn':cam_fn, 'cat_id':cat_id, 
                    'shape_id':shape_id, 'density_fn':density_fn}
                datalist.append(data)

        # prepare all the camara params and the transmat
        for i in range(len(datalist)):
            data = datalist[i]
            cam_fn = data['cam_fn']
            cam_list, transmat_list = self.read_rendering_meta(cam_fn)
            data['camparams'] = cam_list
            data['transmats'] = transmat_list
            datalist[i] = data
        
        self.datalist = datalist
        self.datasize = len(self.datalist)
        print('Finished loading the %s dataset: %d data.'%(status, self.datasize))

    def __len__(self):
        return self.datasize

    def __getitem__(self, index):
        data = self.datalist[index]
        cat_id, shape_id = data['cat_id'], data['shape_id']
        rgb_fn, normal_fn, transmat_list = data['rgba_dir'], data['normal_dir'], data['transmats']
        h5_fn, density_fn = data['h5_fn'], data['density_fn']

        # randomly select a view
        if self.config.train_augmented:
            avai_cam_views = self.config.augmented_cam_views
            rand_cam_id = random.choice(avai_cam_views)
        else:
            rand_cam_id = random.randint(0,self.viewnum-1)
        rgb_image = self.read_rgba_image(rgb_fn, rand_cam_id)
        normal_image = self.read_normal_image(normal_fn, rand_cam_id)
        transmat = transmat_list[rand_cam_id]

        # read ground-truth point-value
        f = h5py.File(h5_fn, 'r')
        samples = f['pc_sdf_sample']
        cent = f['norm_params'][:3]
        cent = np.expand_dims(cent,axis=0)
        scale = f['norm_params'][3]
        points = samples[:,:3]*scale
        points = points + np.repeat(cent, points.shape[0],axis=0)
        values = samples[:,3]*10.0 #(samples[:,3]-0.003)*10.0
        gradients = f['pc_gradients'][:] # the gradients are used to classify "front-side" and "back-side" points
        
        f = h5py.File(density_fn, 'r')
        densities = f['density'][:] # density is used for sampling
        # sample a subset of points
        sampled_pids = self.sampling(values, densities, MODE=self.sampling_mode)
        points = points[sampled_pids]
        values = values[sampled_pids]
        gradients = gradients[sampled_pids] 

        # return the data
        points = torch.tensor(points).float()
        values = torch.tensor(values).float().unsqueeze(1)
        gradients = torch.tensor(gradients).float()
        transmat = torch.tensor(transmat).float()
        rgb_image = torch.tensor(rgb_image).float()
        normal_image = torch.tensor(normal_image).float()
        mc_image = torch.cat([rgb_image, normal_image], dim=2)
        mc_image = mc_image.permute(2,0,1)
        scale = torch.tensor(scale).float() 
        return points, values, gradients, mc_image, transmat, scale

    def get_testdata(self, cat_id, shape_id, cam_id):
    
        if self.config.train_augmented:
            rgb_fn = self.config.image_dir + cat_id + '/' + shape_id + "/"
        else:
            rgb_fn = self.config.image_dir + cat_id + '/' + shape_id + "/easy/"
        normal_fn = self.config.normal_dir + cat_id + '_easy/' + shape_id + '/'
        cam_fn = self.config.cam_dir + cat_id + "/" + shape_id + '/easy' + '/rendering_metadata.txt'

        # read data
        if(os.path.exists(cam_fn) and os.path.exists(rgb_fn)):
            rgba_image = self.read_rgba_image(rgb_fn, cam_id)
            normal_image = self.read_normal_image(normal_fn, cam_id)
            _, transmat_list = self.read_rendering_meta(cam_fn)
            transmat = transmat_list[cam_id]

            transmat = torch.tensor(transmat).float()
            rgba_image = torch.tensor(rgba_image).float()
            rgba_image = rgba_image.permute(2,0,1)
            normal_image = torch.tensor(normal_image).float()
            normal_image = normal_image.permute(2,0,1)
            return rgba_image, normal_image, transmat
        else:
            return None, None, None

    def sampling(self, values, densities, MODE='weighted'):
        if(MODE=='random'):
            sampled_pids = np.random.randint(values.shape[0], size=self.num_points)
            return sampled_pids
        elif(MODE=='weighted'):
            half_sampling_num = int(self.num_points/2)
            pos_inds = np.argwhere(values>0)
            neg_inds = np.argwhere(values<0)
            if(len(pos_inds)<=self.num_points/5 or len(neg_inds)<=self.num_points/5):
                # if there is too few points in one side...
                # just random sampling here
                sampled_pids = np.random.randint(values.shape[0], size=self.num_points)
                return sampled_pids
            else:
                pos_probs = densities[pos_inds]/np.sum(densities[pos_inds])
                pos_probs = np.squeeze(pos_probs, axis=1)
                pos_probs = np.squeeze(pos_probs, axis=1)
                neg_probs = densities[neg_inds]/np.sum(densities[neg_inds])
                neg_probs = np.squeeze(neg_probs, axis=1)
                neg_probs = np.squeeze(neg_probs, axis=1)

                if(pos_inds.shape[0]>half_sampling_num):
                    sampled_pos_inds = np.random.choice(pos_inds.shape[0], size=half_sampling_num, replace=False, p=pos_probs)
                else:
                    sampled_pos_inds = np.random.choice(pos_inds.shape[0], size=half_sampling_num-pos_inds.shape[0], replace=False, p=pos_probs)
                    another = np.array([i for i in range(pos_inds.shape[0])])
                    sampled_pos_inds = np.concatenate((sampled_pos_inds, another), axis=0)
                sampled_pos_inds = pos_inds[sampled_pos_inds]
                if(neg_inds.shape[0]>half_sampling_num):
                    sampled_neg_inds = np.random.choice(neg_inds.shape[0], size=half_sampling_num, replace=False, p=neg_probs)
                else:
                    sampled_neg_inds = np.random.choice(neg_inds.shape[0], size=half_sampling_num-neg_inds.shape[0], p=neg_probs)
                    another = np.array([i for i in range(neg_inds.shape[0])])
                    sampled_neg_inds = np.concatenate((sampled_neg_inds, another), axis=0)
                sampled_neg_inds = neg_inds[sampled_neg_inds]

                sampled_pids = np.concatenate((sampled_pos_inds, sampled_neg_inds), axis=0)
                sampled_pids = np.squeeze(sampled_pids, axis=1)
        else:
            # if it's not uniform sampling or weighted sampling
            print("Sampling mode error!!")
            exit()
        return sampled_pids

    def read_rendering_meta(self, rendering_metadata_filename):
        cam_list = []
        transmat_list = []

        fid = open(rendering_metadata_filename, 'r')
        line = fid.readline()
        while(line):
            # read parameters from the file
            startid = line.index('[')
            endid = line.index(']')
            data = line[startid+1:endid]            
            data = data.split(',')
            cam = [float(d) for d in data]
            # compute the transformation matrix
            K, RT = CamUtils.getBlenderProj(cam[0], cam[1], cam[3], img_w=224, img_h=224)
            trans_mat = np.linalg.multi_dot([K, RT, rot_mat])        
            trans_mat_right = np.transpose(trans_mat)
            cam_list.append(cam)
            transmat_list.append(trans_mat_right)
            line = fid.readline()
        # return all the cameras in the file
        return cam_list, transmat_list

    def read_rgba_image(self, img_dir, cam_id):
        img_fn = img_dir + str(cam_id).zfill(2) + '.png'
        image = cv2.imread(img_fn)[:,:,:3]
        image = image/255.0
        return image

    def read_normal_image(self, normal_dir, cam_id):
        # Note the Blender rendered normals are sRGB color mode.
        # The pre-processed images "xx_mnormal.png" are in RGB color mode.
        normal_fn = normal_dir + str(cam_id).zfill(2) + '_mnormal.png'
        normals = cv2.imread(normal_fn)
        normals = (normals/255.0-0.5)*2.0
        return normals

    def read_shape_ids_from_file(self, filename):
        shape_id_list = []
        fid = open(filename, 'r')
        lines = fid.readlines()
        lines = [l.strip('\n') for l in lines]
        return lines

    """ compute the transformation matrix from RT """
    def process_transmat(self, RT):
        F_MM = 35.  # Focal length
        SENSOR_SIZE_MM = 32.
        PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
        RESOLUTION_PCT = 100.
        SKEW = 0.
        img_w, img_h = 224,224

        # Calculate intrinsic matrix.
        scale = RESOLUTION_PCT / 100
        f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
        f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
        # print('f_u', f_u, 'f_v', f_v)
        u_0 = img_w * scale / 2
        v_0 = img_h * scale / 2
        K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

        trans_mat = np.linalg.multi_dot([K, RT, rot_mat])        
        trans_mat_right = np.transpose(trans_mat)
        return trans_mat_right