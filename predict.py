# https://github.com/lucidrains/denoising-diffusion-pytorch/issues/185

import numpy as np
from PIL import Image
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from funcs import draw_mulit_images_in_one


if __name__ == "__main__":
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # gen model
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8))

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    # load weight
    runs = "200"
    state_dict: dict = torch.load(f"results/flowers/200000/model-{runs}.pt")
    print(state_dict.keys())  # ['step', 'model', 'opt', 'ema', 'scaler', 'version']
    diffusion.load_state_dict(state_dict["model"])
    diffusion.to(device)
    diffusion.eval()

    return_all_timesteps = True

    # sample image
    with torch.inference_mode():
        y = diffusion.sample(batch_size=36, return_all_timesteps=return_all_timesteps)
    print(y.size())
    y1 = y.cpu().numpy()
    y1 = np.array(y1 * 255, dtype=np.uint8)

    if not return_all_timesteps:
        y1 = np.transpose(y1, [0, 2, 3, 1])  # [B, C, H, W] -> [B, H, W, C]
        image = Image.fromarray(draw_mulit_images_in_one(y1, width_repeat=6))
        image.save(f"model-{runs}.png")
    else:
        y1 = np.transpose(y1, [0, 1, 3, 4, 2])  # [B, S, C, H, W] -> [B, S, H, W, C]
        # 保存结果图片
        image = Image.fromarray(draw_mulit_images_in_one(y1[:, -1], width_repeat=6))
        image.save(f"model-{runs}.png")
        # 保存序列图片
        for i, yy in enumerate(y1):
            image = Image.fromarray(draw_mulit_images_in_one(yy, width_repeat=16))
            image.save(f"model-{runs}-{i}.png")
