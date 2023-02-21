import argparse
import PIL
import torch
import numpy as np
from PIL import Image
from itertools import islice
from scipy.ndimage import binary_dilation

from ldm.util import instantiate_from_config

def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to render"
    )
    parser.add_argument(
        "--init_img",
        type=str,
        nargs="?",
        help="path to the input image"
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        nargs="?",
        default=512,
        help="image resolution (will resize image)",
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )
    parser.add_argument(
        "--save_in",
        action='store_true',
        help="save input images to outdir",
    )
    parser.add_argument(
        "--sweep_tuples",
        action='store_true',
        help="sweep parameters are seen as attached tuples instead of sweeping through all combinations",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        nargs="*",
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        nargs='*',
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given image",
    )
    parser.add_argument(
        "--max_imgs",
        type=int,
        nargs='?',
        default=1000000,
        help="max number of images from input",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size (number of images in single forward pass)",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inpainting-inference.yaml",
        # default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/sd-v1-5-inpainting.ckpt",
        # default="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs='*',
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--scale",
        type=float,
        nargs='*',
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--strength",
        type=float,
        nargs='*',
        default=1.0,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument(
        "--mask",
        type=str,
        nargs="?",
        help="path to mask"
    )
    parser.add_argument(
        "--down_N_in",
        type=float,
        nargs='*',
        default=1.0,
        help="ILVR downsampling factor inside mask"
    )
    parser.add_argument(
        "--down_N_out",
        type=float,
        nargs='*',
        default=1.0,
        help="ILVR downsampling factor outside mask"
    )
    parser.add_argument(
        "--T_in",
        type=float,
        nargs='*',
        default=0.0,
        help="strength of ILVR inside mask (in [0.0, 0.1])"
    )
    parser.add_argument(
        "--T_out",
        type=float,
        nargs='*',
        default=0.0,
        help="strength of ILVR outside mask (in [0.0, 0.1])"
    )
    parser.add_argument(
        "--blend_pix",
        type=int,
        nargs='*',
        default=0,
        help="number of pixels for mask smoothing"
    )
    parser.add_argument(
        "--pixel_cond_space",
        action='store_true',
        help="if enabled, uses pixel space for ILVR conditioning, otherwise uses latent space (doesn't work due to"
             "first stage decoder bug)",
    )
    parser.add_argument(
        "--repaint_start",
        type=float,
        nargs='*',
        default=0.0,
        help="Use RePaint (https://arxiv.org/pdf/2201.09865.pdf) for conditioning for (r*time_steps steps), (r in [0.0, 0.1])",
    )
    parser.add_argument(
        "--ilvr_x0",
        action='store_true',
        help="perform ILVR in x_0 space instead of x_t space",
    )
    parser.add_argument(
        "--mask_dilate",
        type=int,
        nargs='*',
        default=0,
        help="Dilate mask to contain larger region (# pixels to dilate)",
    )
    parser.add_argument(
        "--shadow",
        action='store_true',
        help="adjust mask for shadow generation",
    )

    parser.add_argument(
        "--prompt_in",
        type=str,
        nargs="?",
        help="control prompt inside masked region"
    )
    parser.add_argument(
        "--prompt_out",
        type=str,
        nargs="?",
        help="control prompt outside masked region"
    )
    parser.add_argument(
        "--prompt_amplifier_in",
        type=float,
        nargs="*",
        default=1.,
        help="amplifier (w) of paint-by-word inside masked region"
    )
    parser.add_argument(
        "--prompt_amplifier_out",
        type=float,
        nargs="*",
        default=1.,
        help="amplifier (w) of paint-by-word outside masked region"
    )
    return parser

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_img(path, image_size=None):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    r = image_size / max(w, h, image_size) if image_size is not None else 1.
    w, h = map(lambda x: int(r * x), (w, h))
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def load_mask(path, size, dilate, shadow=False):
    mask = Image.open(path).convert("L")
    mask = mask.resize(size, resample=PIL.Image.Resampling.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    if shadow:
        mask = add_shadow(mask)
    mask = mask[None, None]
    mask[mask < 0.05] = 0
    mask[mask >= 0.05] = 1
    if dilate > 0:
        mask = binary_dilation(mask, iterations=dilate).astype(np.float32)
    mask = torch.from_numpy(mask)
    return mask

def add_shadow(mask, width=30):
    shadow_mask = np.zeros_like(mask)
    for object_val in np.unique(mask):
        if object_val == 0:
            continue
        u, d = np.where(np.any(mask == object_val, axis=1))[0][[0, -1]]
        l, r = np.where(np.any(mask == object_val, axis=0))[0][[0, -1]]
        u, d = max(d - width, 0), min(d + width, mask.shape[0])
        l, r = max(l - width, 0), min(r + width, mask.shape[1])
        shadow_mask[u:d, l:r] = object_val
    return shadow_mask