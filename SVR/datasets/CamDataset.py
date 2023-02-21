import os
import h5py
import random
import numpy as np

import torch
from torch.utils import data 
import SVR.datasets.DISN_CamThing as CamUtils
import cv2

class CamDataset(data.Dataset):
    def __init__(self, config, status):
        self.catlist = config.catlist
        self.viewnum = config.viewnum
        self.config = config
        self.world_matrix = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        #self.num_points = 2048

        # read the shape ids from the files
        shape_id_list_from_files = []
        cat_id_list_from_files = []
        for cat_id in self.catlist:
            filename = config.data_dir + cat_id + '_' + status + '.lst'
            shape_ids = self.read_shape_ids_from_file(filename)
            cat_ids = [cat_id for _ in shape_ids]
            shape_id_list_from_files = shape_id_list_from_files + shape_ids
            cat_id_list_from_files = cat_id_list_from_files + cat_ids

        # check the existence of the data files to form the dataset
        datalist = []
        transmat_list = []
        for i in range(len(shape_id_list_from_files)):
            cat_id = cat_id_list_from_files[i]
            shape_id = shape_id_list_from_files[i]
            # each data needs "rgba_image, normal_image, edge_image"
            if config.train_augmented:
                rgba_dir = config.image_dir + cat_id + '/' + shape_id + '/'
            else:
                rgba_dir = config.image_dir + cat_id + '/' + shape_id + '/easy/'
            cam_fn = config.cam_dir + cat_id + '/' + shape_id + '/easy'+'/rendering_metadata.txt'
            h5_fn = config.h5_dir + cat_id + '/' + shape_id + '/data.h5'
            # check the existence of the files
            rgba_existence = os.path.exists(rgba_dir) #We assume all the images (rgba, normal, edge) exist when the rgba_dir exists.
            h5_existence = os.path.exists(h5_fn)
            # add the data into the dataset
            if(rgba_existence and h5_existence):
                data = {'rgba_dir':rgba_dir, 'cat_id':cat_id, 'shape_id':shape_id, 'h5_fn':h5_fn}
                transmats = self.read_transmats(cam_fn)
                transmat_list.append(transmats)
                datalist.append(data)
        
        self.datalist = datalist
        self.transmat_list = transmat_list
        self.datasize = len(self.datalist)
        self.world_matrix = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        
        print('Finished loading the %s dataset: %d data.'%(status, self.datasize))

    def read_shape_ids_from_file(self, filename):
        shape_id_list = []
        fid = open(filename, 'r')
        lines = fid.readlines()
        lines = [l.strip('\n') for l in lines]
        return lines

    def read_transmats(self, cam_fn):
        fid = open(cam_fn, 'r')
        line = fid.readline()
        cam_list = []
        RT_list = []
        while(line):
            startid = line.index('[')
            endid = line.index(']')
            data = line[startid+1:endid]            
            data = data.split(',')
            cam = []
            for d in data:
                cam.append(float(d))
            K, RT = CamUtils.getBlenderProj(cam[0], cam[1], cam[3], img_w=224, img_h=224)
            RT = np.transpose(RT)
            RT_list.append(RT)
            cam_list.append(cam)
            line = fid.readline()
        return RT_list
    
    def read_rgba_image(self, img_dir, cam_id):
        img_fn = img_dir + str(cam_id).zfill(2) + '.png'
        image = cv2.imread(img_fn)[:,:,:3]
        image = image/255.0
        return image

    def __getitem__(self, index):
        # read the points and values
        transmats = self.transmat_list[index]

        data = self.datalist[index]
        cat_id = data['cat_id']
        shape_id = data['shape_id']
        rgba_dir = data['rgba_dir']
        h5_fn = data['h5_fn']

        f = h5py.File(h5_fn, 'r')
        samples = f['pc_sdf_sample']
        cent = f['norm_params'][:3]
        cent = np.expand_dims(cent,axis=0)
        scale = f['norm_params'][3]
        points = samples[:,:3]*scale
        samplings = points + np.repeat(cent, points.shape[0],axis=0)
        
        # read the images 
        
        if self.config.train_augmented:
            avai_cam_views = self.config.augmented_cam_views
            rand_cam_id = random.choice(avai_cam_views)
        else:
            rand_cam_id = random.randint(0,self.viewnum-1)
        rgba_image = self.read_rgba_image(rgba_dir, rand_cam_id)
        transmat = transmats[rand_cam_id]

        # return the data
        samplings = torch.tensor(samplings).float()
        transmat = torch.tensor(transmat).float()
        rgba_image = torch.tensor(rgba_image).float()
        rgba_image = rgba_image.permute(2,0,1)
        rgba_image = rgba_image[:3,:,:]
        return rgba_image, transmat, samplings

    def __len__(self):
        return self.datasize
