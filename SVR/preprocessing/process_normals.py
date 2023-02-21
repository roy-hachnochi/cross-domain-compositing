from __future__ import division
import cv2, os
import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage.filters import maximum_filter as maxf2D

"""
Convert the image from SRGB color space to linear RGB color space. 
The input srgb should be in range [0,1]
"""
def sRGB_to_linear(srgb):
    gamma = ((srgb + 0.055) / 1.055)**2.4
    scale = srgb / 12.92
    return np.where(srgb>0.04045, gamma, scale)

"""
Convert the linear RGB normal map to the unit normal vectors.
The input normal_map is in range [-0.5,0.5].
The output normal_map is in range [-1,1].
The output norm_map is the norm of each vector in the normal map.
"""
def unit_normal_map(normal_map):
    scale = np.sqrt(normal_map[:,:,0]**2 + normal_map[:,:,1]**2 + normal_map[:,:,2]**2)
    scale = scale[:,:,np.newaxis]
    normal_map = normal_map/np.repeat(scale,3,axis=2)
    return normal_map

if __name__ == "__main__":
    # This is all ShapeNet categories rendered in DISN dataset
    # catlist = ['03001627', '02691156', '02828884', '02933112', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04530566','02958343', '04401088']
    
    # Alternatively, we can choose to only process the category of interest
    catlist = ['04256520']

    data_dir = './SVR/data/'
    normal_dir = data_dir + 'normal/'
    output_dir = data_dir + 'normal_processed/'

    for cat in catlist:
        model_ids = os.listdir(normal_dir+cat)
        shape_num = len(model_ids)
        for i in range(shape_num):
            mid = model_ids[i]
            input_folder = normal_dir + cat + '/' + mid + '/easy/' 
            output_folder = output_dir + cat + '_easy/' + mid + '/'

            if(not os.path.exists(output_folder)):
                os.makedirs(output_folder)
               
            for viewid in range(36):
                normalfn = input_folder + str(viewid).zfill(2) + '.png'
                output_normalfn = output_folder + str(viewid).zfill(2) + '_mnormal.png'

                srgb = cv2.imread(normalfn)
                linear_normal = sRGB_to_linear(srgb/255.0)-0.5
                unit_normal = unit_normal_map(linear_normal)
                gt_nmap = unit_normal*0.5+ 0.5
        
                cv2.imwrite(output_normalfn, gt_nmap*255.0)
            print('processing: %d/%d, %s done.' % (i, shape_num, mid))