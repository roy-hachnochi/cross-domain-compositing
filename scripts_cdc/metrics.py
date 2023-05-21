import lpips
from tqdm import tqdm
import numpy as np
import torch
import PIL
from PIL import Image
import argparse
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import clip
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import csv

# ======================================================================================================================
# From InstructPix2Pix: https://github.com/timothybrooks/instruct-pix2pix/blob/main/metrics/clip_similarity.py
class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text) -> torch.Tensor:
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image):  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(self, image_0, image_1, text_0, text_1):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim_i0_t0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_i1_t0 = F.cosine_similarity(image_features_1, text_features_0)
        sim_i0_t1 = F.cosine_similarity(image_features_0, text_features_1)
        sim_i1_t1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_i0_t0, sim_i1_t0, sim_i0_t1, sim_i1_t1, sim_direction, sim_image

    def image_similarity(self, image_0, image_1):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_image


class MaskedLPIPS(nn.Module):
    def __init__(self, net='alex', use_mask=True):
        super().__init__()
        # VGG - closer to "traditional" perceptual loss, alex- best forward scores
        self.lpips = lpips.LPIPS(net=net, spatial=use_mask)  # best forward scores
        self.use_mask = use_mask

    def forward(self, im1, im2, mask=None):
        loss = self.lpips(im1, im2)
        if self.use_mask:
            if mask is None:
                mask = 1.
            loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax  # top, left, bottom, right

# ======================================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate metrics for multiple/single prediction image/directory. Saves all metric\'s mean to'
                    'single csv file. Optional to also save metrics of all files to additional csv\'s.')
    parser.add_argument('--pred', type=str, nargs='+', required=True, help='path to the prediction image/dir')
    parser.add_argument('--gt', type=str, required=True, help='path to the ground truth image/dir')
    parser.add_argument('--mask', type=str, required=True, help='path to the FG/BG mask image/dir')
    parser.add_argument('--text_in', type=str, required=True, help='path to file containing input domain texts')
    parser.add_argument('--text_out', type=str, required=True, help='path to file containing output domain texts')
    parser.add_argument('--outfile', type=str, required=True, help='path to the output file')
    parser.add_argument('--clip_ver', type=str, nargs='*', default='ViT-L/14@336px', help='CLIP version to use for metrics')
    parser.add_argument('--save_all', '-s', action='store_true', help='save metrics for all files as well as mean and STD')

    opt = parser.parse_args()

    # Inputs
    pred_dirs = opt.pred if isinstance(opt.pred, list) else [opt.pred]
    gt_dir = opt.gt
    mask_dir = opt.mask
    texts_in_file = opt.text_in
    texts_out_file = opt.text_out
    outfile = opt.outfile
    clip_vers = opt.clip_ver if isinstance(opt.clip_ver, list) else [opt.clip_ver]
    save_all = opt.save_all

    # Prepare metrics functions
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lpips_metric = MaskedLPIPS(use_mask=False).to(device)
    masked_lpips_metric = MaskedLPIPS(use_mask=True).to(device)

    # Prepare data loop
    print(f"Reading ground truth from {gt_dir}.")
    gt_list = [os.path.join(gt_dir, filename) for filename in sorted(os.listdir(gt_dir))] if os.path.isdir(
        gt_dir) else [gt_dir]
    print(f"Reading masks from {mask_dir}.")
    mask_list = [os.path.join(mask_dir, filename) for filename in sorted(os.listdir(mask_dir))] if os.path.isdir(
        mask_dir) else [mask_dir]
    print(f"Reading input domains from {texts_in_file}.")
    with open(texts_in_file) as f:
        texts_in = f.readlines()
    print(f"Reading output domains from {texts_out_file}.")
    with open(texts_out_file) as f:
        texts_out = f.readlines()

    # Calculate metrics
    outdir = os.path.dirname(outfile)
    f_all = open(outfile, 'w')
    writer_all = csv.writer(f_all)
    writer_all.writerow(['CLIP ver', 'prediction', 'PSNR in', 'LPIPS in', 'PSNR out', 'LPIPS out', 'PSNR', 'SSIM',
                         'LPIPS', 'CLIP direction similarity', 'CLIP similarity improvement'])

    for clip_ver in clip_vers:
        clip_metric = ClipSimilarity(name=clip_ver).to(device)
        for pred_dir in pred_dirs:
            print(f"Reading predictions from {pred_dir}.")
            pred_list = [os.path.join(pred_dir, filename) for filename in
                         sorted(os.listdir(pred_dir))] if os.path.isdir(pred_dir) else [pred_dir]
            pred_name = os.path.basename(os.path.abspath(pred_dir))
            if save_all:
                f = open(os.path.join(outdir, pred_name + '_' + clip_ver.replace(os.path.sep, '') + '_metrics.csv'), 'w')
                writer = csv.writer(f)
                writer.writerow(['name', 'PSNR in', 'LPIPS in', 'PSNR out', 'LPIPS out', 'PSNR', 'SSIM', 'LPIPS',
                                 'CLIP direction similarity', 'CLIP similarity improvement'])

            # Metrics
            psnr_in_ = []
            lpips_in_ = []
            psnr_out_ = []
            lpips_out_ = []
            psnr_ = []
            ssim_ = []
            lpips_ = []
            clip_sim_dir_ = []
            clip_imp_sim_ = []
            for pred_file, gt_file, mask_file, text_in, text_out in tqdm(
                    zip(pred_list, gt_list, mask_list, texts_in, texts_out)):
                img0 = np.array(Image.open(pred_file).convert('RGB')).astype(np.float32) / 127.5 - 1.
                img1 = Image.open(gt_file).convert('RGB')
                mask = Image.open(mask_file).convert("L")
                img1 = img1.resize(img0.shape[:-1][::-1], resample=PIL.Image.Resampling.LANCZOS)
                img1 = np.array(img1).astype(np.float32) / 127.5 - 1.
                mask = mask.resize(img0.shape[:-1][::-1], resample=PIL.Image.Resampling.NEAREST)
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

                psnr_in_.append(peak_signal_noise_ratio(mask * img0, mask * img1))
                lpips_in_.append(masked_lpips_metric(img0_lpips, img1_lpips, mask_torch).item())
                psnr_out_.append(peak_signal_noise_ratio((1. - mask) * img0, (1. - mask) * img1))
                lpips_out_.append(masked_lpips_metric(img0_lpips, img1_lpips, 1. - mask_torch).item())
                psnr_.append(peak_signal_noise_ratio(img0, img1))
                ssim_.append(structural_similarity(img0, img1, channel_axis=2))
                lpips_.append(lpips_metric(img0_lpips, img1_lpips).item())
                sim_gt_tin, sim_pred_tin, sim_gt_tout, sim_pred_tout, sim_dir, sim_im = clip_metric(
                    img1_clip[:, :, bbox_slice[0], bbox_slice[1]], img0_clip[:, :, bbox_slice[0], bbox_slice[1]],
                    [text_in], [text_out])
                clip_sim_dir_.append(sim_dir.item())
                clip_imp_sim_.append(sim_pred_tout.item() / sim_gt_tout.item())
                if save_all:
                    writer.writerow([name, psnr_in_[-1], lpips_in_[-1], psnr_out_[-1], lpips_out_[-1], psnr_[-1],
                                     ssim_[-1], lpips_[-1], clip_sim_dir_[-1], clip_imp_sim_[-1]])

            # Mean and STD
            metrics = [psnr_in_, lpips_in_, psnr_out_, lpips_out_, psnr_, ssim_, lpips_, clip_sim_dir_, clip_imp_sim_]
            mean = ['mean']
            std = ['STD']
            for metric in metrics:
                mean.append(np.mean(metric))
                std.append(np.std(metric))
            if save_all:
                writer.writerow(mean)
                writer.writerow(std)
                f.close()
            writer_all.writerow([clip_ver, pred_name] + mean[1:])
    f_all.close()
