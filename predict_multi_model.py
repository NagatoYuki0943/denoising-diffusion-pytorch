# https://github.com/lucidrains/denoising-diffusion-pytorch/issues/185

import numpy as np
from PIL import Image
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from funcs import draw_mulit_images_in_one
import os


# 遍历全部模型,生成图片
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

    # 是否使用ema model
    use_ema = False

    model_dir = r"results/flowers/100000"
    model_list = os.listdir(model_dir)
    model_list = [model for model in model_list if model.endswith("pt")]

    for model in model_list:
        print(f"model: {model}")

        # load weight
        state_dict: dict = torch.load(os.path.join(model_dir, model))

        if not use_ema:
            diffusion.load_state_dict(state_dict["model"])
        else:
            ema_dict: dict = state_dict["ema"]
            ema_keys = ema_dict.keys()
            ema_model = {}
            for key in ema_keys:
                if "ema_model." in key:
                    ema_model[key[10:]] = ema_dict[key]
            diffusion.load_state_dict(ema_dict)

        diffusion.to(device)
        diffusion.eval()

        # sample image
        with torch.inference_mode():
            y = diffusion.sample(batch_size=36, return_all_timesteps=False)
        y1 = y.cpu().numpy()
        y1 = np.array(y1 * 255, dtype=np.uint8)

        y1 = np.transpose(y1, [0, 2, 3, 1])  # [B, C, H, W] -> [B, H, W, C]
        image = Image.fromarray(draw_mulit_images_in_one(y1, width_repeat=6))
        image.save(os.path.join(model_dir, model[:-2] + "png"))
