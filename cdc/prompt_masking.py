import torch
from torch import einsum
import numpy as np
from transformers import CLIPTokenizer
from einops import rearrange, repeat

from ldm.modules.attention import default, exists


def get_word_inds(text, words, tokenizer):
    split_text = text.split(" ")
    if type(words) is int:
        words = [words]
    elif type(words) is str:
        words = words.split(" ")
    if type(words[0]) is str:
        words = [i for i, word in enumerate(split_text) if word in words]
    out = []
    if len(words) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in words:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

def cross_attention_masks(mask=None, tokens_in=None, tokens_out=None, w_in=1., w_out=1., max_tokens=77):
    if mask is not None:
        mask = mask[:, 0, :, :]
        b, h, w = mask.shape
        token_mask = torch.zeros((b, max_tokens, h, w)).to(mask)
        if tokens_in is not None:
            for token in tokens_in:
                token_mask[:, token, ...] = mask * w_in
        if tokens_out is not None:
            for token in tokens_out:
                token_mask[:, token, ...] = (1 - mask) * w_out
    else:
        token_mask = None
    return token_mask

def register_masked_cross_attention(model, token_mask=None, stop_masking=0):
    """
    Replaces all cross-attention layers in model with new forward that implements paint-by-word for localized token
    enhancement.
    """
    def get_forward(self):
        def forward(x, context=None, mask=None):
            h = self.heads

            q = self.to_q(x)
            context, t, beta = default(context, (x, 0, 0))
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            if exists(token_mask) and t.min().item() >= stop_masking:
                b = token_mask.shape[0]
                latent_size = int(sim.shape[1] ** 0.5)  # using -b*h as indexing makes it work in classifier-free guidance case (batch*2)
                token_mask_r = torch.nn.functional.interpolate(token_mask, size=latent_size)
                token_mask_r = rearrange(token_mask_r, 'b T h w -> b (h w) T')
                token_mask_r = repeat(token_mask_r, 'b ... -> (b h) ...', h=h)
                beta = repeat(beta[-b:], 'b ... -> (b h) 1 1 ...', h=h)
                wA = torch.log(1 + beta) * sim.max() * token_mask_r  # paint by word scheduling weight
                sim[-b*h:] = sim[-b*h:] + wA.type(sim.dtype)  # paint by word (only for cond samples)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

        return forward

    def register_ca(name_, net_, count):
        if name_ == 'attn2' and net_.__class__.__name__ == 'CrossAttention':
            net_.forward = get_forward(net_)
            return count + 1
        elif hasattr(net_, 'named_children'):
            for name__, net__ in net_.named_children():
                count = register_ca(name__, net__, count)
        return count

    cross_att_count = 0
    for name, net in model.model.named_children():
        cross_att_count += register_ca(name, net, 0)

def masked_cross_attention(model, prompts, words_in, words_out, mask, tokenizer, w_in=0., w_out=0., stop_masking=0):
    if exists(mask) and (exists(words_in) or exists(words_out)):
        token_mask = []
        for i, prompt in enumerate(prompts):
            tokens_in = get_word_inds(prompt, words_in, tokenizer) if exists(words_in) else list()
            tokens_out = get_word_inds(prompt, words_out, tokenizer) if exists(words_out) else list()
            token_mask.append(cross_attention_masks(mask[i:i+1], tokens_in, tokens_out, w_in, w_out))
        token_mask = torch.concat(token_mask, dim=0)
        register_masked_cross_attention(model, token_mask, stop_masking)


if __name__ == "__main__":
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    print(get_word_inds("A photograph of a sofa in a living room", ["living", "room"], tokenizer))
