from tqdm import tqdm
import numpy as np
import torch
import PIL
from PIL import Image
import argparse
import os
from skimage.metrics import peak_signal_noise_ratio
from itertools import product

from scripts_cdc.metrics import ClipSimilarity, MaskedLPIPS, get_bbox

# ======================================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate metrics for a single prediction with multiple configurations. Saves results to np'
                    'dictionary.')
    parser.add_argument('--pred', type=str, required=True, help='path to the prediction dir')
    parser.add_argument('--filename', type=str, required=True, help='name of the image to test')
    parser.add_argument('--gt', type=str, required=True, help='path to the ground truth image')
    parser.add_argument('--mask', type=str, required=True, help='path to the FG/BG mask image')
    parser.add_argument('--text_in', type=str, required=True, help='input image text prompt')
    parser.add_argument('--text_out', type=str, required=True, help='output image text prompt')
    parser.add_argument('--outdir', type=str, required=True, help='path to the output dir')
    parser.add_argument('--clip_ver', type=str, nargs='*', default='ViT-L/14@336px', help='CLIP version to use for metrics')

    opt = parser.parse_args()

    # Inputs
    pred_dir = opt.pred
    filename = opt.filename
    gt_file = opt.gt
    mask_file = opt.mask
    text_in = opt.text_in
    text_out = opt.text_out
    outdir = opt.outdir
    clip_vers = opt.clip_ver if isinstance(opt.clip_ver, list) else [opt.clip_ver]
    os.makedirs(outdir, exist_ok=True)

    # Prepare metrics functions
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    masked_lpips_metric = MaskedLPIPS(use_mask=True).to(device)

    # Prepare data loop
    print(f"Reading ground truth from {gt_file}.")
    img1_pil = Image.open(gt_file).convert('RGB')
    print(f"Reading masks from {mask_file}.")
    mask_pil = Image.open(mask_file).convert('L')

    # Initialize
    print(f"Reading predictions from {pred_dir}.")
    pred_list = [dir for dir in sorted(os.listdir(pred_dir)) if (os.path.isdir(os.path.join(pred_dir, dir)) and '_' in dir)]
    params_dict = {}
    for dir in pred_list:
        parts = dir.split('_')
        for i in range(0, len(parts), 2):
            if parts[i] in params_dict.keys():
                params_dict[parts[i]].add(parts[i + 1])
            else:
                params_dict[parts[i]] = {parts[i + 1]}
    size = []
    param_names = list(params_dict.keys())
    for key in param_names:
        params_dict[key] = sorted(list(params_dict[key]))
        size.append(len(params_dict[key]))

    # Calculate metrics:
    for clip_ver in clip_vers:
        clip_metric = ClipSimilarity(name=clip_ver).to(device)
        psnr_in = np.zeros(size)  # PSNR inside mask
        lpips_in = np.zeros(size)  # LPIPS inside mask
        dir_sim = np.zeros(size)  # CLIP directional similarity inside mask
        sim_imp = np.zeros(size)  # CLIP similarity improvement inside mask
        for conf in tqdm(product(*[range(l) for l in size])):
            dir_name = '_'.join([param_names[i] + '_' + params_dict[param_names[i]][conf[i]] for i in range(len(conf))])
            img0 = np.array(Image.open(os.path.join(pred_dir, dir_name, filename)).convert('RGB')).astype(np.float32) / 127.5 - 1.
            img1 = img1_pil.resize(img0.shape[:-1][::-1], resample=PIL.Image.Resampling.LANCZOS)
            img1 = np.array(img1).astype(np.float32) / 127.5 - 1.
            mask = mask_pil.resize(img0.shape[:-1][::-1], resample=PIL.Image.Resampling.NEAREST)
            mask = np.array(mask).astype(np.float32) / 255.

            img0_clip = (torch.from_numpy(img0).to(device)[None, ...].permute(0, 3, 1, 2) + 1.) / 2  # to [0, 1] + (B, C, H, W)
            img1_clip = (torch.from_numpy(img1).to(device)[None, ...].permute(0, 3, 1, 2) + 1.) / 2  # to [0, 1] + (B, C, H, W)
            img0_lpips = torch.from_numpy(img0).to(device)[None, ...].permute(0, 3, 1, 2)   # (B, C, H, W)
            img1_lpips = torch.from_numpy(img1).to(device)[None, ...].permute(0, 3, 1, 2)   # (B, C, H, W)
            mask_torch = torch.from_numpy(mask).to(device)[None, ...]  # (B, H, W)
            mask = mask[..., None]
            t, l, b, r = get_bbox(mask)
            bbox_slice = (slice(t, b+1), slice(l, r+1))
            name = os.path.splitext(os.path.basename(gt_file))[0]

            psnr_in[conf] = peak_signal_noise_ratio(mask * img0, mask * img1)
            lpips_in[conf] = masked_lpips_metric(img0_lpips, img1_lpips, mask_torch).item()
            sim_gt_tin, sim_pred_tin, sim_gt_tout, sim_pred_tout, sim_dir, sim_im = clip_metric(
                img1_clip[:, :, bbox_slice[0], bbox_slice[1]], img0_clip[:, :, bbox_slice[0], bbox_slice[1]],
                [text_in], [text_out])
            dir_sim[conf] = sim_dir.item()
            sim_imp[conf] = sim_pred_tout.item() / sim_gt_tout.item()

        name = os.path.splitext(os.path.basename(filename))[0]
        metrics = {'psnr_in': psnr_in, 'lpips_in': lpips_in, 'dir_sim': dir_sim, 'sim_imp': sim_imp}
        outpath = os.path.join(outdir, name + '_' + clip_ver.replace(os.path.sep, '') + '.npy')
        print(f'Saved metrics to {outpath}')
        np.save(outpath, metrics)
