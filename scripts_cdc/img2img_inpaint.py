"""make variations of input image"""

import os, sys
from itertools import product
import torch
import numpy as np
from omegaconf import OmegaConf, ListConfig
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything

from ldm.models.diffusion.ddim import DDIMSampler
from scripts_cdc.img2img_utils import create_argparser, chunk, load_img, load_mask, load_model_from_config
from cdc.prompt_masking import masked_cross_attention

def load_data(img_path, mask_path, prompt_path, max_imgs, default_prompt, batch_size, mask_dilate, image_size=None, shadow=False):
    print(f"reading images from {img_path}")
    im_list = [os.path.join(img_path, filename) for filename in sorted(os.listdir(img_path))] if os.path.isdir(
        img_path) else [img_path]
    n_imgs = len(im_list)
    if mask_path is not None:
        print(f"reading masks from {mask_path}")
        mask_list = [os.path.join(mask_path, filename) for filename in sorted(os.listdir(mask_path))] if os.path.isdir(
            mask_path) else [mask_path]
        assert(len(mask_list) == n_imgs)

    if not prompt_path:
        prompt = default_prompt
        assert default_prompt is not None
        prompt = [prompt] * n_imgs
    else:
        print(f"reading prompts from {prompt_path}")
        with open(prompt_path, "r") as f:
            prompt = f.read().splitlines()
    assert(len(prompt) == n_imgs)
    prompts = list(chunk(prompt, batch_size))

    img = []
    mask = []
    filename = []
    b = 0
    for i in range(1, min(n_imgs, max_imgs) + 1):
        filename.append(os.path.basename(im_list[i - 1]).split(".")[0])
        img.append(load_img(im_list[i - 1], image_size))
        if mask_path is not None:
            mask.append(load_mask(mask_list[i - 1], tuple(img[-1].shape[-2:][::-1]), mask_dilate, shadow))
        if i % batch_size == 0 or i == n_imgs:
            img = torch.concat(img, dim=0)
            mask = torch.concat(mask, dim=0) if mask_path is not None else None
            masked_img = (1 - mask) * img if mask_path is not None else img
            batch = {"image": img, "mask": mask, "masked_image": masked_img}
            prompt = prompts[b]
            yield batch, prompt, filename
            img, mask, filename = [], [], []
            b += 1

def load_config():
    parser = create_argparser()
    opt, unknown = parser.parse_known_args()
    sampling_conf = OmegaConf.create({'sampling': vars(opt)})
    cli = OmegaConf.from_dotlist(unknown)
    cli = {key.replace('--', ''): val for key, val in cli.items()}
    config = OmegaConf.load(f"{opt.config}")
    config = OmegaConf.merge(sampling_conf, config, cli)
    return config

def load_model_and_sampler(config):
    sampling_conf = config.sampling
    model = load_model_from_config(config, f"{sampling_conf.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if sampling_conf.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        # sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    return model, sampler

def img_inpaint(config, model, sampler, img_res = 512, need_index = True):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sampling_conf = config.sampling
    sweep = {}
    for k in iter(sampling_conf):
        if isinstance(sampling_conf[k], ListConfig):
            if len(sampling_conf[k]) > 1:
                sweep[k] = [(k, x) for x in sampling_conf[k]]
            else:
                sampling_conf[k] = sampling_conf[k][0]
    nrow = sampling_conf.nrow if sampling_conf.nrow > 0 else sampling_conf.n_samples
    precision_scope = autocast if sampling_conf.precision == "autocast" else nullcontext

    os.makedirs(sampling_conf.outdir, exist_ok=True)
    outpath = sampling_conf.outdir
    with open(os.path.join(outpath, 'run_command'), 'w') as f:
        f.write(" ".join(f"\"{i}\"" if " " in i else i for i in sys.argv))
    OmegaConf.save(config, os.path.join(outpath, 'config.yaml'))

    print('Sweeping through: {}'.format({key: [val[1] for val in sweep[key]] for key in sweep}))
    if sampling_conf.sweep_tuples:
        runs = zip(*sweep.values())
        n_exp = len(next(iter(sweep.values())))
    else:
        runs = product(*sweep.values())
        n_exp = np.prod([len(l) for l in sweep.values()])
    print(f'Total of {n_exp} experiments')
    for exp_i, prod in enumerate(runs):
        print("===================================================")
        print(f'Running with {prod}')
        for param, val in prod:
            sampling_conf[param] = val

        data_loader = load_data(sampling_conf.init_img, sampling_conf.mask, sampling_conf.from_file,
                                sampling_conf.max_imgs, sampling_conf.prompt, sampling_conf.batch_size,
                                sampling_conf.mask_dilate, sampling_conf.image_size, sampling_conf.shadow)

        sample_path = os.path.join(outpath, '_'.join([str(val).replace('_', '') for element in prod for val in element]))
        os.makedirs(sample_path, exist_ok=True)
        OmegaConf.save(config, os.path.join(sample_path, 'config.yaml'))

        assert 0. <= sampling_conf.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        sampler.make_schedule(ddim_num_steps=sampling_conf.ddim_steps, ddim_eta=sampling_conf.ddim_eta, verbose=False)
        repaint_conf = OmegaConf.create({'use_repaint': sampling_conf.repaint_start > 0,
                                         'inpa_inj_time_shift': 1,
                                         'schedule_jump_params': {
                                             't_T': sampling_conf.ddim_steps,
                                             'n_sample': 1,
                                             'jump_length': 10,
                                             'jump_n_sample': 10,
                                             'start_resampling': int(sampling_conf.repaint_start * sampling_conf.ddim_steps),
                                             'collapse_increasing': False}})

        grid_count = 0
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    for batch, prompts, filename in tqdm(data_loader, desc="Data"):
                        if (sampling_conf.seed is not None) and (sampling_conf.seed >= 0):
                            seed_everything(sampling_conf.seed)
                        img = batch["image"].to(device)
                        masked_img = batch["masked_image"].to(device)
                        mask = batch["mask"].to(device)
                        batch_size = masked_img.shape[0]

                        init_latent = model.get_first_stage_encoding(model.encode_first_stage(masked_img))  # move to latent space
                        latent_mask = torch.nn.functional.interpolate(mask, size=init_latent.shape[-2:]).to(
                            device) if mask is not None else None
                        c_cat = torch.cat((latent_mask, init_latent), dim=1)

                        uc = None
                        if sampling_conf.scale != 1.0:
                            uc_cross = model.get_learned_conditioning(batch_size * [""])
                            uc = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c_cross = model.get_learned_conditioning(prompts)
                        c = {"c_concat": [c_cat], "c_crossattn": [c_cross]}

                        tokenizer = model.cond_stage_model.tokenizer
                        masked_cross_attention(model, prompts, sampling_conf.prompt_in, sampling_conf.prompt_out,
                                               latent_mask, tokenizer, sampling_conf.prompt_amplifier_in,
                                               sampling_conf.prompt_amplifier_out)

                        for n in trange(sampling_conf.n_samples, desc="Sampling"):
                            # encode (scaled latent)
                            t_enc = int(sampling_conf.strength * sampling_conf.ddim_steps)
                            if t_enc < sampling_conf.ddim_steps:
                                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                            else:  # strength >= 1 ==> use only noise
                                z_enc = torch.randn_like(init_latent)

                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc,
                                                     unconditional_guidance_scale=sampling_conf.scale,
                                                     unconditional_conditioning=uc, ref_image=img, mask=mask,
                                                     down_N_out=sampling_conf.down_N_out,
                                                     down_N_in=sampling_conf.down_N_in, T_out=sampling_conf.T_out,
                                                     T_in=sampling_conf.T_in, blend_pix=sampling_conf.blend_pix,
                                                     pixel_cond_space=sampling_conf.pixel_cond_space,
                                                     repaint=repaint_conf, ilvr_x0=sampling_conf.ilvr_x0)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            if not sampling_conf.skip_save:
                                for i, x_sample in enumerate(x_samples):
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    if need_index:
                                        Image.fromarray(x_sample.astype(np.uint8)).resize((img_res, img_res)).save(
                                            os.path.join(sample_path, f"{filename[i]}_{n:02}.png"))
                                    else:
                                        Image.fromarray(x_sample.astype(np.uint8)).resize((img_res, img_res)).save(
                                            os.path.join(sample_path, f"{filename[i]}.png"))
                            all_samples.append(x_samples)

                    if not sampling_conf.skip_grid:
                        # additionally, save as grid
                        grid = torch.concat(all_samples, 0)
                        grid = make_grid(grid, nrow=nrow)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'grid-{grid_count:04}.png'))
                        grid_count += 1

                    print(f"finished {exp_i + 1}/{n_exp} experiments")

    if sampling_conf.save_in:
        # reload images and save them for reference
        im_dir = os.path.join(outpath, "inputs", "images")
        mask_dir = os.path.join(outpath, "inputs", "masks")
        os.makedirs(im_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        data_loader = load_data(sampling_conf.init_img, sampling_conf.mask, sampling_conf.from_file,
                                sampling_conf.max_imgs, sampling_conf.prompt, sampling_conf.batch_size, 0,
                                sampling_conf.image_size, sampling_conf.shadow)
        for img, mask, prompts, filename in tqdm(data_loader, desc="Data"):
            for i, x_img in enumerate(img):
                x_img = 255. * (x_img.permute(1, 2, 0).cpu().numpy() + 1.) / 2.
                Image.fromarray(x_img.astype(np.uint8)).save(os.path.join(im_dir, f"{filename[i]}.png"))
                if mask is not None:
                    x_mask = 255. * mask[i].permute(1, 2, 0).cpu().numpy().repeat(3, axis=-1)
                    Image.fromarray(x_mask.astype(np.uint8)).save(os.path.join(mask_dir, f"{filename[i]}.png"))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

def main():
    config = load_config()
    model, sampler = load_model_and_sampler(config)
    img_inpaint(config, model, sampler)

if __name__ == "__main__":
    main()
