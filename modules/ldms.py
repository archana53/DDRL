from dataclasses import dataclass

import torch
from diffusers import LDMPipeline
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from torch import nn


@dataclass
class UnconditionalDiffusionModelConfig:
    hf_id: str = "CompVis/ldm-celebahq-256"
    num_diffusion_steps: int = 1000


class UnconditionalDiffusionModel(nn.Module):
    def __init__(self, config):
        super(UnconditionalDiffusionModel, self).__init__()

        self.id = config.hf_id
        self.pipe = LDMPipeline.from_pretrained(self.id)
        self.unet = self.pipe.unet
        self.vqvae = self.pipe.vqvae
        self.num_diffusion_steps = config.num_diffusion_steps
        self.latent = True
        self.up_sample = False
        self.down_sample = True

        # temporary configs
        self.resolution = (256, 256)
        self.up_resolution = (64, 64)
        self.center_crop = True
        self.random_flip = False

        if self.vqvae is not None:
            self.vqvae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(self.resolution)
                if self.center_crop
                else transforms.RandomCrop(self.resolution),
                transforms.RandomHorizontalFlip()
                if self.random_flip
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def get_noisy_samples(self, x, t, is_latent=True):
        if not is_latent:
            x = x[None, :, :, :]
            x = self.vqvae.encode(x).latents

        ddim_scheduler = self.pipe.scheduler
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        noisy_latents = ddim_scheduler.add_noise(x, noise, t)
        return noisy_latents, noise

    def _add_intermediate_feature_hooks(self):
        def hook_fn(module, input, output):
            module.intermediate_features = output

        hooked_modules = []
        for direction in self.scale_direction:
            if direction not in ["up", "down"]:
                self.unet.mid_block.register_forward_hook(hook_fn)
                hooked_modules.append(self.unet.mid_block)
            else:
                for f in self.scales:
                    if direction == "up":
                        self.unet.up_blocks[f].register_forward_hook(hook_fn)
                        hooked_modules.append(self.unet.up_blocks[f])
                    else:
                        self.unet.down_blocks[f].register_forward_hook(hook_fn)
                        hooked_modules.append(self.unet.down_blocks[f])

        self.hooked_modules = hooked_modules

    def set_feature_scales_and_direction(self, scales, scale_direction):
        self.scales = scales
        self.scale_direction = scale_direction
        self._add_intermediate_feature_hooks()
        sample_feature_size = self.get_features(
            torch.randn(1, 3, self.resolution[0], self.resolution[1]),
            torch.LongTensor([10]),
        )[1].size()
        self.feature_size = sample_feature_size

    def get_features(self, x, t):
        if self.latent:  # Convert to latent space
            if len(x.shape) == 3:
                x = x[None, :, :, :]

            x = self.vqvae.encode(x).latents

        noisy_latents, noise_added = self.get_noisy_samples(x, t, is_latent=True)
        noisy_pred = self.unet(noisy_latents, t).sample

        if self.latent:  # get noisy predictions back onto image space
            noisy_pred = self.vqvae.decode(noisy_pred).sample

        # check for tuple returns
        intermediate_features = [m.intermediate_features for m in self.hooked_modules]
        intermediate_features = [
            i[0] if isinstance(i, tuple) else i for i in intermediate_features
        ]
        #upscale to original dimension of the image
        if self.up_sample:
            for i in range(len(intermediate_features)):
                intermediate_features[i] = F.interpolate(
                    intermediate_features[i], size=self.up_resolution, mode="bilinear"
                )
        downsample = []
        if self.down_sample:
            for i in range(len(intermediate_features)):
                kernel = int(intermediate_features[i].shape[3])
                stride = kernel
                intermediate_features[i] = F.avg_pool2d(intermediate_features[i],kernel_size = kernel, stride=stride)
        final_intermediate_features = torch.cat(intermediate_features, axis=1)
        return noisy_pred, final_intermediate_features

    def add_lora_compatibility(self, lora_rank):
        lora_attn_procs = {}
        for name in self.unet.attn_processors:
            print(name)
            block_id = None
            cross_attention_dim = None
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=lora_rank,
            )
        self.unet.set_attn_processor(lora_attn_procs)
        lora_layers = AttnProcsLayers(self.unet.attn_processors)
        return lora_layers

    def get_unet(self):
        # Predicts the noise given an input latent and the time step.
        return self.unet

    def get_vqvae(self):
        # Used to compress the input images into a latent space
        return self.vqvae


if __name__ == "__main__":
    model_config = UnconditionalDiffusionModelConfig()
    model = UnconditionalDiffusionModel(model_config)
    lora_layers = model.add_lora_compatibility(4)
    img = model.image_transforms(Image.open("sample.jpg").convert("RGB"))
    noisy_latents, _ = model.get_noisy_samples(
        img, t=torch.LongTensor([10]), is_latent=False
    )
    noisy_img_pil = transforms.ToPILImage()(noisy_latents.squeeze(0))
    noisy_img_pil.show()
