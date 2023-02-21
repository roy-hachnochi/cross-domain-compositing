import os
from PIL import Image
import numpy as np
from natsort import natsorted
import scipy

def find_mask_margin(mask):
    """Find margin of 2D masks to each boarder, which is the maximum shit distance.

    Args:
        mask (array): 2D mask in array format
    Returns:
        h_margin (list): maximum and minimum horizontal margin shift
        v_margin (list): maximum and minimum vertical margin shift
    """
    w, h = mask.shape[:2]
    # Find horizontal margins
    h_sum = mask.sum(0)
    h_occupancy = np.nonzero(h_sum)
    h_margin = [-h_occupancy[0][0], h - h_occupancy[0][-1]]
    # Find vertical margins
    v_sum = mask.sum(1)
    v_occupancy = np.nonzero(v_sum)
    v_margin = [-v_occupancy[0][0], w - v_occupancy[0][-1]]
    return h_margin, v_margin

def align_masks(mask1_path, mask2_path):
    """Find distance between center of masss of two masks.
    
    Returned displacement represents the distance the mask2 need to travel to match mask1.
    """
    mask1 = np.array(Image.open(mask1_path).resize((224,224)))
    mask2 = np.array(Image.open(mask2_path).resize((224,224)))
    h_margin, v_margin = find_mask_margin(mask2)
    mask1_center = np.array(scipy.ndimage.center_of_mass(mask1))
    mask2_center = np.array(scipy.ndimage.center_of_mass(mask2))
    displacement = mask1_center - mask2_center
    displacement[0] = min(max(v_margin[0], displacement[0]), v_margin[1])
    displacement[1] = min(max(h_margin[0], displacement[1]), h_margin[1])
    return displacement.astype(np.int32)

def process_reference_image(image_path):
    """Binarilise reference masks with a threshold and overwrite the original file.

    Args:
        image_path (str): Path to reference mask

    Returns:
        image (array): Processed reference mask
    """
    image = np.array(Image.open(image_path).convert('L'))
    image = image > 125
    Image.fromarray((image*255).astype(np.uint8)).save(image_path)
    return image

def process_source_image(image_path):
    """Binarilise the source masks with a threshold

    Args:
        image_path (str): Path to the source mask

    Returns:
        image (array): Processed source mask
    """
    image = np.array(Image.open(image_path))[:,:,3]
    image = image > 125 
    return image

def intersection(image1, image2):
    """Intersection between two image vectors."""
    return sum(image1  & image2)

def union(image1, image2):
    """Union between two image vectors."""
    return sum(image1  | image2)

def iou(image1,image2):
    """2D IoU between two image vectors."""
    return intersection(image1, image2)/ union(image1, image2)

def save_batched_masks(source_folder, models_to_compare):
    """Obtain binary mask from rendered images in the source folder and save them to a new directory."""
    for k in range(len(models_to_compare)):
        folders = models_to_compare[k]
        mask_folder = os.path.join(source_folder, folders, "rendered_view")
        save_path = os.path.join(source_folder, folders, "rendered_view_mask")
        os.makedirs(save_path, exist_ok=True)
        for images in os.listdir(mask_folder):
            processed_mask = process_source_image(os.path.join(mask_folder,images))
            Image.fromarray((processed_mask*255).astype(np.uint8)).save(os.path.join(save_path, images))

def shift_image(source_mask, alignment):
    """Shift binary mask with respect to input alignment command."""
    num_dim = len(source_mask.shape)
    template = np.zeros_like(source_mask)
    if num_dim == 2: #mask
        h, w = template.shape[:2]
        template[max(0,alignment[0]): h - abs(min(0, alignment[0])), 
                max(0,alignment[1]): w - abs(min(0, alignment[1]))] = source_mask[abs(min(0, alignment[0])): h - max(0, alignment[0]),
                abs(min(0,alignment[1])): w - max(0, alignment[1])]
    if num_dim == 3: #image
        h, w, c = template.shape
        for i in range(c):
            template[max(0,alignment[0]): h - abs(min(0, alignment[0])), 
                max(0,alignment[1]): w - abs(min(0, alignment[1])), i] = source_mask[abs(min(0, alignment[0])): h - max(0, alignment[0]),
                abs(min(0,alignment[1])): w - max(0, alignment[1]), i]
    return template

def determine_2D_IoU(reference_image_folder, source_image_folder, models_to_compare):
    """Determine average 2D IoU between source masks produced by different models and reference masks."""
    # Reference masks of in the wild images
    reference_masks = os.listdir(reference_image_folder)
    for k in range(len(models_to_compare)):
        folders = models_to_compare[k]
        files = natsorted(os.listdir(source_image_folder + '/' + folders + '/rendered_view'))
        shift_path = os.path.join(source_image_folder, folders, "rendered_view_mask_shifted")
        os.makedirs(shift_path, exist_ok=True)
        iou_sum = 0
        for i in range(len(files)):
            file = files[i]

            # Align masks
            ref_path = os.path.join(reference_image_folder, file.split('.')[0]+'.png')
            source_path = os.path.join(source_image_folder, folders, 'rendered_view_mask', file)
            alignment = align_masks(ref_path, source_path)

            # Shift masks with respect to the geometric mean of the reference mask
            source_mask = np.array(Image.open(source_path).resize((224,224)))
            source_mask = shift_image(source_mask, alignment)
            
            # Save the shifted mask
            Image.fromarray(source_mask.astype(np.uint8)).save(os.path.join(source_image_folder, folders, "rendered_view_mask_shifted", file))
            
            # Reshape the masks to vectors
            source_mask = source_mask.reshape(224*224).astype(bool)
            ref_mask = (np.array(Image.open(os.path.join(reference_image_folder, file.split('.')[0]+'.png')).convert('L'))/255).reshape(224*224).astype(bool)

            ind_iou = iou(ref_mask, source_mask)
            iou_sum += ind_iou
        
        # Determine average IoU
        iou_sum /= len(files)
        print(f"2D IoU of {folders} is {iou_sum}")

def main():
    # Path to the reference images. NOTE: to be changed by the user
    reference_image_folder = "./SVR/result/sofa_sample/sofa_sample_processed_mask"
    # Path to the source images. NOTE: to be changed by the user
    source_image_folder = "./SVR/result/sofa_sample"
    # Model folder name. NOTE: to be changed by the user
    models_to_compare = ['inpaint_050']
    # First extract masks from rendered model silhouette
    save_batched_masks(source_image_folder, models_to_compare)
    # Then determine 2D IoU
    determine_2D_IoU(reference_image_folder, source_image_folder, models_to_compare)

if __name__ == "__main__":
    main()