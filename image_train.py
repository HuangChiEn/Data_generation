"""
Train a diffusion model on images.
"""

import os
import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.unet import UNetModel


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion, vae = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if vae is not None:
        vae.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        is_train=args.is_train,
        use_vae=args.use_vae,
        catch_path=args.catch_path,
        mask_emb=args.mask_emb
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        drop_rate=args.drop_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        training_step=args.training_step,
        vae=vae,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        lr_warmup_steps=0, # !=0 using warmup learning
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        drop_rate=0.0,
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        is_train=True,
        training_step=2000,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
    # if __name__ == "__main__":
    #     path = 'output.txt'
    #     f = open(path, 'w')
    #
    #     unet = UNetModel(
    #         image_size=270,
    #         in_channels=3,
    #         model_channels=256,
    #         out_channels=3,
    #         num_res_blocks=2,
    #         attention_resolutions=(8, 16, 32),
    #         dropout=False,
    #         channel_mult=(1, 1, 2, 4, 4),
    #         num_classes=34,
    #         use_checkpoint=True,
    #         use_fp16=True,
    #         num_heads=8,
    #         num_head_channels=-1,
    #         num_heads_upsample=-1,
    #         use_scale_shift_norm=False,
    #         resblock_updown=False,
    #         use_new_attention_order=False,
    #         mask_emb="resize"
    #     )
    #
    #     print(unet, file=f)
    #     f.close()
