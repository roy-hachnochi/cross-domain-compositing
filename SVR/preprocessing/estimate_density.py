import os
import h5py
import numpy as np
from sklearn.neighbors import KDTree

def collect_density_per_object(anchors, points, values, gridnum=35):
    pos_neighbors = np.zeros((anchors.shape[0],1))
    neg_neighbors = np.zeros((anchors.shape[0],1))

    step = 1.0/gridnum
    tree = KDTree(points)  
    inds = tree.query_radius(anchors, r=step/2.0)
    for p_id in range(anchors.shape[0]):
        nlist = inds[p_id]
        if(len(nlist)>0):
            vs = values[nlist]
            posnum = np.sum(vs<0)
            negnum = np.sum(vs>0)
        else:
            posnum = 0
            negnum = 0
        pos_neighbors[p_id,0] = posnum
        neg_neighbors[p_id,0] = negnum
        
    return pos_neighbors, neg_neighbors

def estimate_density(source_filename, target_filename):
    # read the source data: point-value pairs
    f = h5py.File(source_filename, 'r')
    samples = f['pc_sdf_sample']
    cent = f['norm_params'][:3]
    cent = np.expand_dims(cent,axis=0)
    scale = f['norm_params'][3]
    points = samples[:,:3]*scale
    points = points + np.repeat(cent, points.shape[0],axis=0)
    values = samples[:,3]*10.0

    # compute the inner/outer density
    pos_neighbors, neg_neighbors = collect_density_per_object(points, points, values, gridnum=20)
    densities = np.zeros((points.shape[0],1))
    for i in range(points.shape[0]):
        v = values[i]
        if(v<0):
            densities[i] = 1/pos_neighbors[i]
        else:
            densities[i] = 1/neg_neighbors[i]

    nf = h5py.File(target_filename,'w')
    nf['density'] = densities
    nf.close()

if __name__ == "__main__":
    # This is all ShapeNet categories rendered in DISN dataset
    # catlist = ['03001627', '02691156', '02828884', '02933112', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04530566','02958343', '04401088']
    
    # Alternatively, we can choose to only process the category of interest
    catlist = ['04256520']
    
    data_dir = './SVR/data/'
    h5_dir = data_dir + 'SDF_v1/'
    output_dir = data_dir + 'SDF_density/'

    for cat in catlist:
        model_ids = os.listdir(h5_dir+cat)
        shape_num = len(model_ids)
        for i in range(shape_num):
            mid = model_ids[i]
            input_filename = h5_dir + cat + '/' + mid + '/ori_sample.h5' 
            output_folder = output_dir + cat + '/' + mid + '/'
    
            if(not os.path.exists(output_folder)):
                os.makedirs(output_folder)
            output_filename = output_folder + 'density.h5'
            estimate_density(input_filename, output_filename)
            print('processing: %d/%d, %s done.' % (i, shape_num, mid))
        
