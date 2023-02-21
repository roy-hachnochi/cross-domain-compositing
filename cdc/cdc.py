import torch
from torch import nn
from torch.nn import functional as f

import ResizeRight.resize_right as Resizer

def resizer(scale, out_shape=None):
    def resize(img):
        return Resizer.resize(img, scale_factors=scale, out_shape=out_shape)
    return resize

def get_blend_operator_and_mask(mask, blend_pix):
    if mask is None:
        blend_mask = 1  # use only non-masked - normal ILVR
        blend = Identity()
    else:
        blend = BlurOutwards(n_pixels=blend_pix) if blend_pix is not None and blend_pix > 0 else Identity()
        blend_mask = blend(mask)
    return blend_mask, blend

def get_low_pass_operator(down_N, shape=None):
    if down_N is not None:
        down = resizer(1 / down_N)
        up = resizer(down_N, out_shape=shape)
    else:
        down = Identity()
        up = Identity()
    return down, up

def shift_mask(mask, a, b, T, device):
    if mask is None:
        mask = torch.ones((1, 1, 1, 1)).to(device)
    if a is None:
        a = 0
    if b is None:
        b = T
    mask = (b - a) * mask + a  # map values from [0, 1] to [a, b]
    return mask

def decode_linear(latent):
    T = torch.tensor([[0.298, 0.187, -0.158, -0.184],
                      [0.207, 0.286, 0.189, -0.271],
                      [0.208, 0.173, 0.264, -0.473]]).to(latent.device)
    latent = latent.permute((0, 2, 3, 1))
    image = f.linear(latent, T)
    image = image.permute((0, 3, 1, 2))
    return image

def encode_linear(image):
    T = torch.tensor([[1.69810224, -0.02986101, -0.05746497],
                      [-0.28270747,  4.91430525, -3.04784101],
                      [-2.55163474,  2.23158593,  0.0448761],
                      [-0.78083445,  3.02981481, -3.22913725]]).to(image.device)
    image = image.permute((0, 2, 3, 1))
    latent = f.linear(image, T)
    latent = latent.permute((0, 3, 1, 2))
    return latent

# ======================================================================================================================
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, in_tensor, *ignore_args, **ignore_kwargs):
        return in_tensor

class BlurOutwards(nn.Module):
    def __init__(self, n_pixels, smoother='linear'):
        super(BlurOutwards, self).__init__()
        self.n_pixels = n_pixels
        if smoother == 'sigmoid':
            self.smoother = torch.sigmoid(torch.linspace(-5, 5, self.n_pixels + 1))
            self.smoother[0] = 0
            self.smoother[-1] = 1
        elif smoother == 'linear':
            self.smoother = torch.linspace(0, 1, self.n_pixels + 1)
        else:
            raise NotImplementedError(f'\'{smoother}\' smoothing method is not implemented.')
        self.smoother_diff = torch.diff(self.smoother)

    def forward(self, in_tensor):
        strel = torch.ones((3, 3)).to(in_tensor.device)
        out_tensor = torch.zeros_like(in_tensor)
        for i in range(self.n_pixels):
            out_tensor = out_tensor + self.smoother_diff[i] * in_tensor
            in_tensor = self._dilation(in_tensor, strel, origin=(1, 1))
        return out_tensor

    def _dilation(self, image, strel, origin=(0, 0)):
        """
        Credit: https://stackoverflow.com/a/66496234/12384018
        """
        # first pad the image to have correct unfolding; here is where the origins is used
        image_pad = f.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1])
        # Unfold the image to be able to perform operation on neighborhoods
        image_unfold = f.unfold(image_pad, kernel_size=strel.shape)
        # Flatten the structural element since its two dimensions have been flatten when unfolding
        strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
        # Perform the greyscale operation; sum would be replaced by rest if you want erosion
        sums = image_unfold * strel_flatten  # + in source, more convenient to use * for max operation
        # Take maximum over the neighborhood
        result, _ = sums.max(dim=1)
        # Reshape the image to recover initial shape
        return torch.reshape(result, image.shape)