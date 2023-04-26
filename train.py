from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


if __name__ == "__main__":
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1'            # L1 or L2
    )

    trainer = Trainer(
        diffusion,
        r'../datasets/intel-image-classification-scenery/train&val/glacier',
        train_batch_size = 16,
        train_lr = 8e-5,
        train_num_steps = 100000,         # total training steps (Sharing answer from the question "How many iterations do I need?")[https://github.com/lucidrains/denoising-diffusion-pytorch/issues/121]
        save_and_sample_every = 1000,     # eva&save&sample steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                      # turn off mixed precision, maybe cause NaN on 1080ti
        fp16 = False,
        calculate_fid = True,             # whether to calculate fid during training
        results_folder = "results/intel-image-classification-scenery/glacier/100000",
    )

    trainer.train()
