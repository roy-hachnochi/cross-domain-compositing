from PIL import Image
import os

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def check_experiment(exp_name, path, const_axes=None, ignore=None):
    if not os.path.isdir(os.path.join(path, exp_name)):
        return False
    if not ((const_axes is None) or (len(const_axes) == 0) or all([name in subdir for name in const_axes])):
        return False
    return (ignore is None) or (len(ignore) == 0) or all([name not in subdir for name in ignore])

# ======================================================================================================================
im_dir = "/disk2/royha/stable-diffusion/outputs/scribbles/more"
col_axis = 'Tin'
row_axis = ''
const_axes = ['Nin_1', 'repaintstart_0.6']
ignore = ['Tin_0.4', 'Tin_0.5', 'Tin_0.7']
im_i = 0
# ======================================================================================================================

out_filename = f"summary_{im_i}.jpg" if (const_axes is None) or len(const_axes) == 0 else f"summary_{im_i}_{'_'.join(const_axes)}.jpg"
out_filename = os.path.join(im_dir, out_filename)
const_axes += [col_axis, row_axis]

# load images
im_list = {}
for subdir in os.listdir(im_dir):
    if check_experiment(subdir, im_dir, const_axes, ignore):
        files = sorted(os.listdir(os.path.join(im_dir, subdir)))
        files = list(filter(lambda f: ".png" in f or ".jpg" in f, files))
        x = float(subdir.split(row_axis)[1].split('_')[1]) if row_axis else 0
        y = float(subdir.split(col_axis)[1].split('_')[1]) if col_axis else 0
        if x not in im_list:
            im_list[x] = {}
        filename = files[im_i]
        im_list[x][y] = Image.open(os.path.join(im_dir, subdir, filename))

# make grid
imgs = []
for x in sorted(im_list.keys()):  # rows
    for y in sorted(im_list[x].keys()):  # cols
        imgs.append(im_list[x][y])
grid = image_grid(imgs, len(im_list), len(im_list[list(im_list.keys())[0]]))
grid.save(out_filename)
