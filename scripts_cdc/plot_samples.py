# from PIL import Image
# import PIL
# import os
#
#
# def image_grid(imgs, rows, cols):
#     assert len(imgs) == rows * cols
#     w, h = imgs[0].size
#     grid = Image.new('RGB', size=(cols * w, rows * h))
#     for i, img in enumerate(imgs):
#         grid.paste(img, box=(i % cols * w, i // cols * h))
#     return grid
#
# # ======================================================================================================================
# im_dir = "/disk2/royha/stable-diffusion/outputs/blending/bus_Tout_Nout"
# in_dir = '/home/royha/stable-diffusion/sample_data/Bus/images'
# out_filename = f"summary_0_5_7_11.png"
# out_filename_input = f"input_0_5_7_11.png"
# Ts = ['0.0', '0.2', '0.4', '0.6', '0.8']
# Ns = ['1', '1', '1', '1']
# ims = [0, 5, 7, 11]
# # ======================================================================================================================
#
# im_list = {}
# for t in Ts:
#     for im, N in zip(ims, Ns):
#         subdir = f'downNout_{N}_Tout_{t}'
#         if os.path.isdir(os.path.join(im_dir, subdir)):
#             files = sorted(os.listdir(os.path.join(im_dir, subdir)))
#             files = list(filter(lambda f: ".png" in f, files))
#             if im not in im_list:
#                 im_list[im] = {}
#             filename = files[im]
#             im_list[im][t] = Image.open(os.path.join(im_dir, subdir, filename))
#
# imgs = []
# for x in sorted(im_list.keys()):  # rows
#     for y in sorted(im_list[x].keys()):  # cols
#         imgs.append(im_list[x][y])
# grid = image_grid(imgs, len(im_list), len(im_list[list(im_list.keys())[0]]))
# grid.save(os.path.join(im_dir, out_filename))
#
# # save inputs for reference
# files = sorted(os.listdir(os.path.join(in_dir)))
# files = list(filter(lambda f: ".png" in f, files))
# imgs = []
# for i, filename in enumerate(files):
#     if i in ims:
#         img = Image.open(os.path.join(in_dir, filename))
#         img = img.resize((512, 512), resample=PIL.Image.Resampling.LANCZOS)
#         imgs.append(img)
#
# grid = image_grid(imgs, len(imgs), 1)
# grid.save(os.path.join(im_dir, out_filename_input))


from PIL import Image
import PIL
import os


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# ======================================================================================================================
im_dir = "/disk2/royha/stable-diffusion/outputs/CG2Real/outpainting/sofa_all_seed_t0.6"
in_dir = '/disk2/royha/editing-data/scribbles/images'
out_filename = f"summary_compare.png"
out_filename_input = f"input_partial.png"
seeds = ['downNout_1_Tout_0.4_repaintstart_0.8',
         'downNout_1_Tout_0.2_repaintstart_0.0',
         'downNout_2_Tout_0.2_repaintstart_0.8',
         'downNout_2_Tout_0.4_repaintstart_0.8',
         'downNout_2_Tout_0.2_repaintstart_0.6']
ims = [0, 1]
# ims = list(map(lambda x: x + 36 * 4, (0, 4, 8, 12, 16, 20, 24, 28, 32)))
# ims = [12, 32, 36 + 12, 36 + 32, 36 * 2 + 12, 36 * 2 + 32, 36 * 3 + 12, 36 * 3 + 32, 36 * 4 + 12, 36 * 4 + 32]
# ======================================================================================================================

im_list = {}
for seed in seeds:
    for im in ims:
        subdir = f'{seed}'
        if os.path.isdir(os.path.join(im_dir, subdir)):
            files = sorted(os.listdir(os.path.join(im_dir, subdir)))
            files = list(filter(lambda f: ".png" in f, files))
            if seed not in im_list:
                im_list[seed] = {}
            filename = files[im]
            im_list[seed][im] = Image.open(os.path.join(im_dir, subdir, filename))

imgs = []
for x in sorted(im_list['downNout_1_Tout_0.4_repaintstart_0.8'].keys()):  # rows
    for y in sorted(im_list.keys()):  # cols
        imgs.append(im_list[y][x])
grid = image_grid(imgs, len(im_list[list(im_list.keys())[0]]), len(im_list))
grid.save(os.path.join(im_dir, out_filename))

# save inputs for reference
files = sorted(os.listdir(os.path.join(in_dir)))
files = list(filter(lambda f: ".png" in f, files))
imgs = []
for i, filename in enumerate(files):
    if i in ims:
        img = Image.open(os.path.join(in_dir, filename))
        img = img.resize((512, 512), resample=PIL.Image.Resampling.LANCZOS)
        imgs.append(img)

grid = image_grid(imgs, 1, len(imgs))
grid.save(os.path.join(im_dir, out_filename_input))
