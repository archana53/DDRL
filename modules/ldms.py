from dataclasses import dataclass

import torch
from diffusers import LDMPipeline
from PIL import Image
from torchvision import transforms

breakpoint()


@dataclass
class UnconditionalDiffusionModelConfig:
    hf_id: str = "CompVis/ldm-celebahq-256"
    num_diffusion_steps: int = 50


class UnconditionalDiffusionModel:
    def __init__(self, config):
        self.id = config.hf_id
        self.pipe = LDMPipeline.from_pretrained(self.id)
        self.unet = self.pipe.unet
        self.vqvae = self.pipe.vqvae
        self.num_diffusion_steps = config.num_diffusion_steps

        # temporary configs
        self.resolution = (256, 256)
        self.center_crop = True
        self.random_flip = False

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

    def get_noisy_samples(self, imgs, t=10):
        imgs = imgs[None, :, :, :]
        imgs = self.vqvae.encode(imgs).latents
        ddim_scheduler = self.pipe.scheduler
        # Get the noise according the noise schedule and timestep
        noise = torch.randn(imgs.shape, dtype=(torch.float32))
        t = torch.LongTensor([t])
        noisy_latents = ddim_scheduler.add_noise(imgs, noise, t)
        return noisy_latents, noise

    def get_unet(self):
        # Predicts the noise given an input latent and the time step.
        return self.unet

    def get_vqvae(self):
        # Used to compress the input images into a latent space
        return self.vqvae

    def train_unet(self, train_imgs):
        train_imgs = train_imgs[None, :, :, :]
        latents = self.vqvae.encode(train_imgs)

        random_timesteps = torch.randint(
            0, self.num_diffusion_steps, (train_imgs.shape[0],)
        ).long()
        noisy_images, noise_true = self.get_noisy_samples(train_imgs, random_timesteps)
        noise_pred = self.unet(
            noisy_images, random_timesteps
        ).sample  # predict the noise residual

        # Use these predictions for MSE between actual noise and generated noise

        timesteps = torch.randint
        pass


if __name__ == "__main__":
    model_config = UnconditionalDiffusionModelConfig()
    model = UnconditionalDiffusionModel(model_config)
    img = model.image_transforms(Image.open("sample.jpg").convert("RGB"))
    noisy_latents, _ = model.get_noisy_samples(img, t=10)
    noisy_img = model.get_vqvae().decode(noisy_latents).sample
    noisy_img_pil = transforms.ToPILImage()(noisy_img.squeeze(0))
    noisy_img_pil.show()
