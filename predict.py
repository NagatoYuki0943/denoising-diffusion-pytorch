# https://github.com/lucidrains/denoising-diffusion-pytorch/issues/185

import numpy as np
from PIL import Image
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from funcs import draw_mulit_images_in_one


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # gen model
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    # load weight
    state_dict: dict = torch.load("results/intel-image-classification-scenery/100000/model-50.pt")
    print(state_dict.keys()) # ['step', 'model', 'opt', 'ema', 'scaler', 'version']
    diffusion.load_state_dict(state_dict["model"])
    diffusion.to(device)
    # sample image
    diffusion.eval()

    return_all_timesteps = True

    with torch.inference_mode():
        y = diffusion.sample(batch_size=25, return_all_timesteps=return_all_timesteps)
    print(y.size())
    y1 = y.cpu().numpy()
    y1 = np.array(y1 * 255, dtype=np.uint8)

    if not return_all_timesteps:
        y1 = np.transpose(y1, [0, 2, 3, 1])     # [B, C, H, W] -> [B, H, W, C]
        image = Image.fromarray(draw_mulit_images_in_one(y1, width_repeat=5))
        image.save("result.png")
    else:
        y1 = np.transpose(y1, [0, 1, 4, 3, 2])  # [B, S, C, H, W] -> [B, S, H, W, C]
        for i, yy in enumerate(y1):
            image = Image.fromarray(draw_mulit_images_in_one(yy, width_repeat=16))
            image.save(f"{i}.png")
