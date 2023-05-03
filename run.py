import argparse

import torch

from IPython.display import HTML
from PIL import Image
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, logging

from PIL import Image
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import randn_tensor, PIL_INTERPOLATION
from diffusers.models.embeddings import get_timestep_embedding

torch.manual_seed(1)
logging.set_verbosity_error()
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


def init_pipe():
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = pipe.to(device)
    return pipe


def topk(emb, k=128, abs_vals=True, largest=True):
    topk = torch.topk(torch.abs(emb), k) if abs_vals else torch.topk(emb, k, largest=largest)
    topk_vals = torch.index_select(emb, 1, topk.indices.squeeze())
    return torch.zeros_like(emb).scatter(1, topk.indices, topk_vals)


def max_abs(x, y):
    maxs = torch.max(torch.abs(x), torch.abs(y))
    x_signs = (maxs == torch.abs(x)) * torch.sign(x)
    y_signs = (maxs == torch.abs(y)) * torch.sign(y)
    final_signs = x_signs.int() | y_signs.int()
    return maxs * final_signs


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * \
          low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


class Pipe:
    def __init__(self):
        self.pipe = init_pipe()
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.feature_extractor = self.pipe.feature_extractor
        self.image_encoder = self.pipe.image_encoder
        self.image_normalizer = self.pipe.image_normalizer
        self.image_noising_scheduler = self.pipe.image_noising_scheduler

    def pil_to_latent(self, input_im):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        dtype = next(self.vae.parameters()).dtype
        with torch.no_grad():
            latent = self.vae.encode(
                tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device, dtype=dtype) * 2 - 1)  # Note scaling
        return 0.18215 * latent.latent_dist.sample()

    def latents_to_pil(self, latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def prep_embeds(self, emb, noise_level):
        dtype = next(self.image_encoder.parameters()).dtype
        noise = randn_tensor(emb.shape, device=torch_device, dtype=dtype)
        noise_level = torch.tensor([noise_level] * emb.shape[0], device=emb.device)
        emb = self.image_normalizer.scale(emb)
        emb = self.image_noising_scheduler.add_noise(emb, timesteps=noise_level, noise=noise)
        emb = self.image_normalizer.unscale(emb)

        noise_level = get_timestep_embedding(
            timesteps=noise_level, embedding_dim=emb.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
        )

        noise_level = noise_level.to(emb.dtype)
        emb = torch.cat((emb, noise_level), 1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        emb = emb.unsqueeze(1)
        bs_embed, seq_len, _ = emb.shape
        emb = emb.repeat(1, 1, 1)
        emb = emb.view(bs_embed * 1, seq_len, -1)
        emb = emb.squeeze(1)

        return emb

    def embed_image(self, image1, image2, ratio=0.5):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image1, torch.Tensor):
            image1 = self.feature_extractor(images=image1, return_tensors="pt").pixel_values

        if not isinstance(image2, torch.Tensor):
            image2 = self.feature_extractor(images=image2, return_tensors="pt").pixel_values

        image1 = image1.to(device=torch_device, dtype=dtype)
        image1_embeds = self.image_encoder(image1).image_embeds

        image2 = image2.to(device=torch_device, dtype=dtype)
        image2_embeds = self.image_encoder(image2).image_embeds

        combo_pos = slerp(ratio, image1_embeds, image2_embeds)

        image_embeds = self.prep_embeds(combo_pos, 0)

        # negative_prompt_embeds = prep_embeds(combo_neg, 0)
        negative_prompt_embeds = torch.zeros_like(image_embeds)

        image_embeds = torch.cat([negative_prompt_embeds, image_embeds])
        return image_embeds

    def __call__(self, content_image, style_image, num_inference_steps=50, guidance_scale=8, ratio=0.5):
        style_embs = self.embed_image(content_image, style_image)

        prompt = ["character, 3d style, 8k, beautiful, digital painting, artstation, highly detailed, sharp focus"]
        neg_prompt = [
            "repulsive, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"]

        content_encoded = self.pil_to_latent(content_image)
        style_encoded = self.pil_to_latent(style_image)

        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(torch_device))[0]

        uncond_input = self.tokenizer(neg_prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                      truncation=True, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        self.scheduler.set_timesteps(num_inference_steps)

        start_step = 0
        noise = torch.randn_like(content_encoded)
        latents = self.scheduler.add_noise(content_encoded, noise,
                                           timesteps=torch.tensor([self.scheduler.timesteps[start_step]]))
        latents = latents.to(torch_device)

        # Loop
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            if i >= start_step:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        class_labels=style_embs,
                    ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return self.latents_to_pil(latents)[0]


def open_image(url, size=768):
    image = Image.open(url).convert("RGB")
    w, h = image.size
    if h < w:
        h, w = size, size * w // h
    else:
        h, w = size * h // w, size

    image = image.resize((w, h))
    box = ((w - size) // 2, (h - size) // 2, (w + size) // 2, (h + size) // 2)
    return image.crop(box)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('content_url', type=str)
    parser.add_argument('style_url', type=str)
    parser.add_argument('ratio', type=float, default=0.5)

    args = parser.parse_args()

    content_image = open_image(args.content_url)
    style_image = open_image(args.style_url)

    pipe = Pipe()
    output = pipe(content_image, style_image, args.ratio)

    output.save("output.png")
