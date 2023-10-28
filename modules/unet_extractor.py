import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torchvision import transforms

from PIL import Image

from diffusers import LDMPipeline


class ModelExtractor(nn.Module):
    '''
    Module wrapper for extracting intermediate features from a model.
    '''

    def __init__(self, pipe, model, scale_direction, scales, latent=True, upsample=True):
        '''
        Args:
            pipe (LDMPipeline): The pipeline used to generate the noise.\n
            model (nn.Module): The model to extract features from.\n
            scale_direction (list): The direction of the scales to extract features from.\n
            scales (list): The scales to extract features from.\n
            latent (bool): Whether or not the model takes in latents.
            upsample (bool): Whether or not to upsample the output features.
        '''
        super().__init__()
        self.pipe = pipe
        self.model = model
        self.scale_direction = scale_direction
        self.scales = scales
        self.latent = latent
        self.dim_reduce = pipe.vqvae if latent else None
        self.upsample = upsample

    def forward(self, x, t):
        '''
        Args:
            x (torch.Tensor) [batch_size, channels, height, width] : The input tensor.\n
            t (torch.Tensor) [batch_size,]: The timestep to generate noise for. Note that 
            this timestep is a single timestep for each image in the batch. Multiple 
            timesteps for each image in the batch is not supported. It can be simulated by 
            calling this function multiple times with different timesteps or batch same 
            input multiple times with different timesteps.

        Returns:
            noisy_pred (torch.Tensor) [batch_size, channels, height, width] : The predicted noise.\n
            intermediate_features (dict): The intermediate features extracted from the model.\n
            intermediate_features[name] (torch.Tensor) [batch_size, channels, height, width] : The intermediate feature with the name `name`.
        '''

        # Register hook to extract intermediate features
        def hook_fn(module, input, output, scale=0, direction="mid"):
            # Dictionary to store intermediate features
            # the scale is scaling in the UNet and depends on sampling of the input
            # direction is either "up", "down", or "mid"
            intermediate_features[module._get_name() + "_" + str(direction) + "{" + str(scale) + "}"] = output
        intermediate_features = {}

        # Register hooks
        for direction in self.scale_direction:
            # Register hook for mid_block if direction is not up or down
            if direction not in ["up", "down"]:
                # Register hook for mid_block
                self.model.mid_block.register_forward_hook(hook_fn)
            else:
                # Register hook for up_blocks or down_blocks
                for f in self.scales:
                    if direction == "up":
                        self.model.up_blocks[f].register_forward_hook(partial(hook_fn, scale=f, direction=direction))
                    else:
                        self.model.down_blocks[f].register_forward_hook(partial(hook_fn, scale=f, direction=direction))

        # Convert to latent representation if a latent model
        if self.latent:
            x = self.dim_reduce.encode(x).latents
        
        # Generate noise
        noise = torch.randn(x.shape, dtype=(torch.float32)).to(x.device)
        # Forwards pass through scheduler for each image in the batch
        noisy_latents = self.pipe.scheduler.add_noise(x, noise, t)
        # Forwards pass through model to predict noise
        noisy_pred = self.model(noisy_latents, t).sample

        # Convert back to image representation of the noise if a latent model
        if self.latent:
            noisy_pred = self.dim_reduce.decode(noisy_pred).sample

        # Interpolate intermediate features to the same size as the input
        for name, feat in intermediate_features.items():
            # If the feature is from an DownBlock then huggingsface returns a tuple of output and hidden state
            # which is most likely used for the skip connection and thereby we have to only take teh output of
            # the block and as for the up_block we directly get the output so we do not need to check for tuple
            intermediate_features[name] = feat[0] if type(feat) is tuple else feat
            # Interpolate to the same size as the input
            if self.upsample:
                intermediate_features[name] = F.interpolate(intermediate_features[name], size=noisy_latents.shape[-2:], mode='bilinear')

        # Return the predicted noise and the intermediate features
        # Intermediate features are returned as a dictionary with the
        # name of the layer as the key and the feature as the value
        return noisy_pred, intermediate_features


if __name__ == "__main__":
    
    # Load model
    pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")
    unet = pipe.unet

    # Fix input size
    resolution = (256, 256)
    resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    tensor = transforms.ToTensor()

    # Load image
    image = Image.open("sample.png").convert("RGB")
    image = resize(image)
    image = tensor(image)
    image = image.unsqueeze(0)
    image = torch.cat([image, image], dim=0)

    # Extract features
    scales = [2, 3]
    scale_direction = ["down", "up"]
    timestep = 3

    # Create model extractor
    model = ModelExtractor(pipe, unet, scale_direction, scales)

    # Extract features
    # Time step is a single timestep for each image in the batch if
    # there are multiple images i.e batch_size > 1 then the same timestep
    # is used for all images in the batch. Multiple timesteps for each image
    # is created only when a single image is passed in the batch i.e batch_size = 1
    # To simulate multiple timesteps for each image in the batch, call this function
    # multiple times with different timesteps or batch same input multiple times with
    # different timesteps.
    timesteps = torch.arange(1, timestep + 1) if image.shape[0] == 1 else torch.LongTensor([timestep])
    timesteps = timesteps.to(image.device)
    _, features = model(image, timesteps)
