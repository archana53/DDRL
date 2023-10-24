import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from PIL import Image

from diffusers import LDMPipeline

class ModelExtractor(nn.Module):

    def __init__(self, pipe, model, scale_direction, scales, latent=True):
        super().__init__()
        self.pipe = pipe
        self.model = model
        self.scale_direction = scale_direction
        self.scales = scales
        self.latent = latent
        self.dim_reduce = pipe.vqvae if latent else None

    def forward(self, x, t):

        def hook_fn(module, input, output):
            intermediate_features.append(output)
        intermediate_features = []

        for direction in self.scale_direction:
            if direction not in ["up", "down"]:
                self.model.mid_block.register_forward_hook(hook_fn)
            else:
                for f in self.scales:
                    if direction == "up":
                        self.model.up_blocks[f].register_forward_hook(hook_fn)
                    else:
                        self.model.down_blocks[f].register_forward_hook(hook_fn)

        if self.latent:
            x = self.dim_reduce.encode(x).latents
        
        noise = torch.randn(x.shape, dtype=(torch.float32)).to(x.device)
        noisy_latents = self.pipe.scheduler.add_noise(x, noise, t)
        noisy_pred = self.model(noisy_latents, t).sample

        if self.latent:
            noisy_pred = self.dim_reduce.decode(noisy_pred).sample

        intermediate_features = [i[0] for i in intermediate_features]
        for i in range(len(intermediate_features)):
            intermediate_features[i] = F.interpolate(intermediate_features[i], size=noisy_latents.shape[-2:], mode='bilinear')

        return noisy_pred, intermediate_features
    
if __name__ == "__main__":
    
    pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")
    unet = pipe.unet

    resolution = (256, 256)
    resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    tensor = transforms.ToTensor()

    image = Image.open("sample.png").convert("RGB")
    image = resize(image)
    image = tensor(image)
    image = image.unsqueeze(0)
    image = torch.cat([image, image], dim=0)

    scales = [2]
    scale_direction = ["down"]
    timestep = 2

    model = ModelExtractor(pipe, unet, scale_direction, scales)

    timesteps = torch.arange(1, timestep + 1) if image.shape[0] == 1 else torch.LongTensor([timestep])
    timesteps = timesteps.to(image.device)
    _, features = model(image, timesteps)
