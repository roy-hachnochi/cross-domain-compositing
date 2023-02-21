import os
import h5py
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize


def estimate_gradient(source_filename, target_filename):
    # read data
    f = h5py.File(source_filename, 'r')
    samples = f['pc_sdf_sample']
    points = samples[:,:3]
    values = np.expand_dims(samples[:,3], axis=1)

    # query the K-nearest neighbors
    tree = KDTree(points)     
    dists, inds = tree.query(points, k=10)  
    pc_gradients = []
    for kid in inds:
        X = points[kid,:]
        Y = values[kid,:]
        reg = LinearRegression().fit(X, Y)
        gradient = reg.coef_
        gradient = normalize(gradient)
        pc_gradients.append(np.squeeze(gradient))
    pc_gradients = np.array(pc_gradients)
    
    # save target
    pc_sdf_original = f['pc_sdf_original'][:]
    pc_sdf_sample = f['pc_sdf_sample'][:]
    sdf_params = f['sdf_params'][:]
    norm_params = f['norm_params'][:]
    ft = h5py.File(target_filename, 'w')
    ft['pc_sdf_original'] = pc_sdf_original
    ft['pc_sdf_sample'] = pc_sdf_sample
    ft['sdf_params'] = sdf_params
    ft['norm_params'] = norm_params
    ft['pc_gradients'] = pc_gradients
    ft.close()


if __name__ == "__main__":
    # This is all ShapeNet categories rendered in DISN dataset
    # catlist = ['03001627', '02691156', '02828884', '02933112', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04530566','02958343', '04401088']
    
    # Alternatively, we can choose to only process the category of interest
    catlist = ['04256520']

    data_dir = './SVR/data/'
    h5_dir = data_dir + 'SDF_v1/'
    output_dir = data_dir + 'SDF_with_gradient/'

    for cat in catlist:
        model_ids = os.listdir(h5_dir+cat)
        shape_num = len(model_ids)
        for i in range(shape_num):
            mid = model_ids[i]
            input_filename = h5_dir + cat + '/' + mid + '/ori_sample.h5' 
            output_folder = output_dir + cat + '/' + mid + '/'

            if (not os.path.exists(output_folder)):
                os.makedirs(output_folder)

            output_filename = output_folder + 'data.h5'
            estimate_gradient(input_filename, output_filename)
            print('processing: %d/%d, %s done.' % (i, shape_num, mid))
            