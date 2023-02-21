# import os
# import shutil
#
# # ======================================================================================================================
# in_path = "/disk2/royha/stable-diffusion/iHarmony4-dataset/Hday2night"
# data_filename = "/disk2/royha/stable-diffusion/iHarmony4-dataset/Hday2night/Hday2night_test.txt"
# out_path = "/disk2/royha/stable-diffusion/iHarmony4-dataset/Hday2night/test"
# # ======================================================================================================================
#
# images_in_path = os.path.join(in_path, "composite_images")
# masks_in_path = os.path.join(in_path, "masks")
# gt_in_path = os.path.join(in_path, "real_images")
# images_out_path = os.path.join(out_path, "images")
# masks_out_path = os.path.join(out_path, "masks")
# gt_out_path = os.path.join(out_path, "gts")
# os.makedirs(images_out_path, exist_ok=True)
# os.makedirs(masks_out_path, exist_ok=True)
# os.makedirs(gt_out_path, exist_ok=True)
#
# with open(data_filename) as file:
#     filenames = [line.rstrip() for line in file]
#
# for filename in filenames:
#     mask_filename = filename.rsplit("_", 1)[0]
#     gt_filename = filename.split("_")[0]
#     shutil.copyfile(os.path.join(images_in_path, filename), os.path.join(images_out_path, filename))
#     try:
#         shutil.copyfile(os.path.join(masks_in_path, mask_filename + ".png"), os.path.join(masks_out_path, filename))
#     except FileNotFoundError:
#         shutil.copyfile(os.path.join(masks_in_path, mask_filename + ".jpg"), os.path.join(masks_out_path, filename))
#     try:
#         shutil.copyfile(os.path.join(gt_in_path, gt_filename + ".jpg"), os.path.join(gt_out_path, filename))
#     except FileNotFoundError:
#         shutil.copyfile(os.path.join(gt_in_path, gt_filename + ".png"), os.path.join(gt_out_path, filename))



import os
from PIL import Image
import numpy as np

# ======================================================================================================================
in_path = "/disk2/royha/datasets/CG2Real-dataset/04256520-sofa"
subdirs = ["1a384", "1a477", "1a713", "1acdc", "5ad77"]
out_path = "/disk2/royha/datasets/CG2Real-dataset/test/sofa"
# ======================================================================================================================

images_out_path = os.path.join(out_path, "images")
masks_out_path = os.path.join(out_path, "masks")
os.makedirs(images_out_path, exist_ok=True)
os.makedirs(masks_out_path, exist_ok=True)

for subdir in os.listdir(in_path):
    if any(substring in subdir for substring in subdirs):
        prefix = subdir[:5]
        path = os.path.join(in_path, subdir, 'easy')
        files = sorted(os.listdir(path))
        files = list(filter(lambda f: ".png" in f, files))
        for filename in files:
            image_obj = Image.open(os.path.join(path, filename))
            w, h = image_obj.size
            s = 1.5 + 0.5 * np.random.rand()
            w, h = map(lambda x: int(s * x), (w, h))
            image_obj = image_obj.resize((w, h))
            alpha_layers = np.array(image_obj)[:, :, 3]
            mask_obj = (alpha_layers != 0)
            image_obj = np.array(image_obj)[:, :, :3]
            image = np.zeros((512, 512, 3)).astype(np.uint8)
            mask = np.zeros((512, 512, 3)).astype(np.uint8)
            r = np.random.randint(0, 512 - image_obj.shape[0] + 1)
            c = np.random.randint(0, 512 - image_obj.shape[1] + 1)
            image[r:r + image_obj.shape[0], c:c + image_obj.shape[1], :] = image_obj
            mask[r:r + mask_obj.shape[0], c:c + mask_obj.shape[1], :] = mask_obj[:, :, np.newaxis] * 255
            Image.fromarray(image).save(os.path.join(images_out_path, prefix + "-" + filename))
            Image.fromarray(mask).save(os.path.join(masks_out_path, prefix + "-" + filename))
