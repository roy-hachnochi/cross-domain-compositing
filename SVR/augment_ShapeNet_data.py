"""Augment ShapeNet data with inpainting model."""
import os
import SVR.utils as SVR_utils
import scripts_cdc.img2img_inpaint as inpaintlib
import numpy as np
from PIL import Image, ImageOps

def process_image(image_file, save_path):
    """Resize and get mask from input images."""
    image = Image.open(image_file)
    resized_image = image.resize((512,512))
    w, h = resized_image.size
    # Obtain mask
    alpha_layers = np.array(resized_image)[:,:,3]
    mask = alpha_layers != np.zeros((w,h))
    mask = ImageOps.invert(Image.fromarray(mask))
    # Save new image and mask with sorted name
    new_image = Image.fromarray(np.array(resized_image)[:,:,:3])
    image_save_path = os.path.join(save_path, image_file.split('/')[-1])
    mask_save_path = os.path.join(save_path, f"image_mask.png")
    new_image.save(image_save_path)
    mask.save(mask_save_path)
    return image_save_path, mask_save_path


def main():
    """Augment shapenet data with preset cam views and saving folder."""
    # Get SVR model training arguments and image editing arguments
    SVR_config = SVR_utils.get_args()
    cdc_config = inpaintlib.load_config()

    # Specify image editing prompts
    cdc_config.sampling.prompt = SVR_config.prompt

    # Specigy inpainting model path
    cdc_config.sampling.ckpt = "./models/ldm/stable-diffusion-v1/sd-v1-5-inpainting.ckpt"

    # NOTE: Tout is controlling the object's condition strength. 1 for pure preservation.
    cdc_config.sampling.T_out = SVR_config.augmentation_strength
    
    # Other parameters that are mostly remain unchanged
    cdc_config.sampling.strength = 1.0
    cdc_config.sampling.ddim_steps = 50
    cdc_config.sampling.T_in = 0
    cdc_config.sampling.down_N_in = 1
    cdc_config.sampling.down_N_out = 1
    cdc_config.sampling.blend_pix = 0
    cdc_config.sampling.max_imgs = 1
    cdc_config.sampling.from_file = False
    cdc_config.sampling.skip_grid = True

    SVR_config = SVR_utils.get_args()
 
    # Path to save augmented data
    save_folder_identifier = str(int(cdc_config.sampling.T_out * 100)).zfill(3)
    save_path = "./SVR/data/image_augmented_inpaint_" + save_folder_identifier
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Load model and sampler at once
    model, sampler = inpaintlib.load_model_and_sampler(cdc_config)

    # Path to rendered ShapeNet images
    shapeNet_data = "./SVR/data/image"

    # Category id of the target ShapeNet object
    cat_id = SVR_config.catlist
    
    # Camera view points to be augmented
    cam_views_num = SVR_config.augmented_cam_views
    cam_view = [str(x).zfill(2) for x in cam_views_num]

    # A directory to temporarily store interim products, such as resized images and masks.
    tmp_save_path = "./SVR/data/tmp_storage"
    if not os.path.exists(tmp_save_path):
        os.mkdir(tmp_save_path)

    for cat in cat_id:
        instances = os.listdir(os.path.join(shapeNet_data, cat))
        for instance in instances:
            # For each model instance, augment the easy views, which are centered in the middle.
            instance_path = os.path.join(shapeNet_data, cat, instance ,'easy')

            # Specify saving directory w.r.t. model id
            cdc_config.sampling.outdir = os.path.join(save_path, cat, instance)
        
            if not os.path.exists(cdc_config.sampling.outdir):
                os.makedirs(cdc_config.sampling.outdir,exist_ok=True)
            
            # Augment each camera view
            for cam in cam_view:
                file_name = os.path.join(instance_path, cam+".png")
                result_name = os.path.join(cdc_config.sampling.outdir, cam+'.png')
                # If an augmentation is already made for this camera view, skip this loop.
                if not os.path.isfile(result_name):                  
                    # Process the input images and temporarily save them in the tmp save folder.
                    image_save_path, mask_save_path = process_image(file_name, tmp_save_path)
                    cdc_config.sampling.init_img = image_save_path
                    cdc_config.sampling.mask = mask_save_path

                    # Augment image 
                    inpaintlib.img_inpaint(cdc_config, model, sampler, img_res=224, need_index=False)

if __name__ == "__main__":
    main()